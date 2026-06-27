import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from prepare import MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, FRAME_DIM, TRAIN_BIN, VAL_BIN, Dataloader, evaluate_loss

# ---------------------------------------------------------------------------
# Hyperparameters (agent modifies these)
# ---------------------------------------------------------------------------
N_LAYER = 6
N_HEAD = 8
N_EMBD = 256
TOKEN_EMBD_DIM = 64
BATCH_SIZE = 64          # Batch size per GPU (effective batch size = 128)
LEARNING_RATE = 8e-4     # Slightly adjusted for larger batch size
WEIGHT_DECAY = 0.01

# ---------------------------------------------------------------------------
# GPT Model Components
# ---------------------------------------------------------------------------
class FrameEmbedding(nn.Module):
    def __init__(self, vocab_size, token_embd_dim, n_embd, frame_dim=FRAME_DIM):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, token_embd_dim)
        self.proj = nn.Linear(frame_dim * token_embd_dim, n_embd, bias=False)
        
    def forward(self, x):
        B, L, F = x.size()
        emb = self.wte(x) # (B, L, 128, token_embd_dim)
        emb = emb.view(B, L, F * emb.size(-1)) # (B, L, 128 * token_embd_dim)
        return self.proj(emb) # (B, L, n_embd)

class FrameHead(nn.Module):
    def __init__(self, n_embd, token_embd_dim, frame_dim=FRAME_DIM):
        super().__init__()
        self.proj = nn.Linear(n_embd, frame_dim * token_embd_dim, bias=False)
        self.frame_dim = frame_dim
        self.token_embd_dim = token_embd_dim
        
    def forward(self, x, wte_weight):
        B, L, C = x.size()
        features = self.proj(x) # (B, L, 128 * token_embd_dim)
        features = features.view(B, L, self.frame_dim, self.token_embd_dim)
        logits = torch.matmul(features, wte_weight.T) # (B, L, 128, 1024)
        return logits

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class SwiGLUMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(2 * (4 * n_embd) / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = SwiGLUMLP(n_embd)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, token_embd_dim, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wfe = FrameEmbedding(vocab_size, token_embd_dim, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)]),
            ln_f = RMSNorm(n_embd)
        ))

        self.lm_head = FrameHead(n_embd, token_embd_dim)
        self.block_size = block_size
        
    def forward(self, idx):
        device = idx.device
        t = idx.size(1)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        x = self.transformer.wfe(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Tied weights classifier
        wte_weight = self.transformer.wfe.wte.weight
        logits = self.lm_head(x, wte_weight)
        return logits

def get_lr_multiplier(elapsed, total_time):
    warmup_frac = 0.05
    if elapsed < total_time * warmup_frac:
        return elapsed / (total_time * warmup_frac)
    else:
        progress = (elapsed - total_time * warmup_frac) / (total_time * (1 - warmup_frac))
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))

# ---------------------------------------------------------------------------
# Training Execution
# ---------------------------------------------------------------------------
def train():
    # Setup DDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        device = 'cuda'
        master_process = True

    torch.manual_seed(42 + ddp_rank)
    torch.cuda.manual_seed(42 + ddp_rank)
    torch.set_float32_matmul_precision("high")
    
    # Initialize model
    model = GPT(VOCAB_SIZE, TOKEN_EMBD_DIM, N_EMBD, N_HEAD, N_LAYER, MAX_SEQ_LEN).to(device)
    model = torch.compile(model, mode="reduce-overhead")
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Setup data loaders
    train_loader = Dataloader(TRAIN_BIN, BATCH_SIZE, MAX_SEQ_LEN)
    val_loader = Dataloader(VAL_BIN, BATCH_SIZE, MAX_SEQ_LEN)
    
    # Optimizer and FP16 GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)
    scaler = torch.cuda.amp.GradScaler()
    
    if master_process:
        print(f"Starting training (DDP: {ddp}, Time Budget: {TIME_BUDGET}s)...")
    t_start = time.time()
    step = 0
    
    model.train()
    while True:
        t0 = time.time()
        x, y = train_loader.get_batch()
        
        # Update learning rate based on time
        elapsed = time.time() - t_start
        lrm = get_lr_multiplier(elapsed, TIME_BUDGET)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE * lrm
            
        optimizer.zero_grad()
        
        # FP16 mixed precision forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        step += 1
        elapsed = time.time() - t_start

        
        if master_process and step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
            
        if elapsed >= TIME_BUDGET:
            break
            
    if master_process:
        print("Training finished. Evaluating...")
        # Unwrap model for evaluation if using DDP
        raw_model = model.module if ddp else model
        val_loss, val_bpt, comp_ratio = evaluate_loss(raw_model, val_loader)
        
        print("\n--- RESULTS ---")
        print(f"val_loss: {val_loss:.6f}")
        print(f"val_bpt: {val_bpt:.6f}")
        print(f"comp_ratio: {comp_ratio:.6f}")
        print(f"num_params: {sum(p.numel() for p in raw_model.parameters()):,}")

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
