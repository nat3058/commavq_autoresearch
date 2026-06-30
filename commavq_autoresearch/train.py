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
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

TOKEN_EMBD_DIM = 64
BATCH_SIZE = 128          # Batch size per GPU (effective batch size = 256)
LEARNING_RATE = 4.0e-3
WEIGHT_DECAY = 0.03
N_GROUPS = 8


# ---------------------------------------------------------------------------
# GPT Model Components
# ---------------------------------------------------------------------------
class FrameEmbedding(nn.Module):
    def __init__(self, vocab_size, token_embd_dim, n_embd, frame_dim=FRAME_DIM):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, token_embd_dim)
        self.conv = nn.Conv2d(token_embd_dim, token_embd_dim, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(8, token_embd_dim)
        self.proj_groups = nn.ModuleList([
            nn.Linear((frame_dim // N_GROUPS) * token_embd_dim, n_embd // N_GROUPS, bias=False)
            for _ in range(N_GROUPS)
        ])
        self.n_groups = N_GROUPS
        self.frame_dim = frame_dim
        
    def forward(self, x):
        B, L, frame_dim = x.size()
        emb = self.wte(x) # (B, L, 128, token_embd_dim)
        # Reshape to 2D spatial grid (B * L, token_embd_dim, 8, 16)
        emb_grid = emb.view(B * L, 8, 16, emb.size(-1)).permute(0, 3, 1, 2)
        emb_grid = emb_grid + F.silu(self.gn(self.conv(emb_grid)))
        # Reshape back to flat sequence
        emb = emb_grid.permute(0, 2, 3, 1).contiguous().view(B, L, self.n_groups, -1)
        
        # Grouped projection
        outputs = []
        for i, proj in enumerate(self.proj_groups):
            outputs.append(proj(emb[:, :, i, :]))
        return torch.cat(outputs, dim=-1) # (B, L, n_embd)


class FrameHead(nn.Module):
    def __init__(self, n_embd, token_embd_dim, frame_dim=FRAME_DIM):
        super().__init__()
        self.proj_groups = nn.ModuleList([
            nn.Linear(n_embd // N_GROUPS, (frame_dim // N_GROUPS) * token_embd_dim, bias=False)
            for _ in range(N_GROUPS)
        ])
        self.conv1 = nn.Conv2d(token_embd_dim, token_embd_dim, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, token_embd_dim)
        self.conv2 = nn.Conv2d(token_embd_dim, token_embd_dim, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, token_embd_dim)
        self.conv3 = nn.Conv2d(token_embd_dim, token_embd_dim, kernel_size=3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(8, token_embd_dim)
        self.frame_dim = frame_dim
        self.token_embd_dim = token_embd_dim
        self.n_groups = N_GROUPS
        
    def forward(self, x, wte_weight):
        B, L, C = x.size()
        x_chunks = torch.chunk(x, self.n_groups, dim=-1)
        outputs = []
        for i, proj in enumerate(self.proj_groups):
            outputs.append(proj(x_chunks[i]))
        features = torch.cat(outputs, dim=-1) # (B, L, 128 * token_embd_dim)
        
        # Reshape to 2D spatial grid (B * L, token_embd_dim, 8, 16)
        features = features.view(B * L, 8, 16, self.token_embd_dim).permute(0, 3, 1, 2)
        
        # Spatial coordination refinement (3 residual layers)
        features = features + F.silu(self.gn1(self.conv1(features)))
        features = features + F.silu(self.gn2(self.conv2(features)))
        features = features + F.silu(self.gn3(self.conv3(features)))
        
        # Reshape back to flat tokens
        features = features.permute(0, 2, 3, 1).contiguous().view(B, L, self.frame_dim, self.token_embd_dim)
        wte_weight_t = wte_weight.t().contiguous()
        logits = torch.matmul(features, wte_weight_t) # (B, L, 128, 1024)
        return logits



class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, q, k, seq_len):
        device = q.device
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1).to(device)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1).to(device)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        self.rotary_emb = RotaryEmbedding(n_embd // n_head)
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rotary_emb(q, k, T)
        
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

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, n_layer=6):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = SwiGLUMLP(n_embd)
        self.scale = 1.0 / math.sqrt(2.0 * n_layer)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) * self.scale
        x = x + self.mlp(self.ln_2(x)) * self.scale
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, token_embd_dim, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wfe = FrameEmbedding(vocab_size, token_embd_dim, n_embd),
            ln_f = nn.LayerNorm(n_embd)
        ))
        self.block = Block(n_embd, n_head, block_size, n_layer)
        self.lm_head = FrameHead(n_embd, token_embd_dim)
        self.block_size = block_size
        self.n_layer = n_layer
        
    def forward(self, idx):
        x = self.transformer.wfe(idx)
        for _ in range(self.n_layer):
            x = self.block(x)
        x = self.transformer.ln_f(x)
        
        # Tied weights classifier
        wte_weight = self.transformer.wfe.wte.weight
        logits = self.lm_head(x, wte_weight)
        return logits



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
    
    # Enable cuDNN autotuning for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    # Initialize model
    model = GPT(VOCAB_SIZE, TOKEN_EMBD_DIM, N_EMBD, N_HEAD, N_LAYER, MAX_SEQ_LEN).to(device)
    
    # Always compile model for maximum steps/sec
    use_compile = True
    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")
        
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Setup data loaders
    train_loader = Dataloader(TRAIN_BIN, BATCH_SIZE, MAX_SEQ_LEN)
    val_loader = Dataloader(VAL_BIN, BATCH_SIZE, MAX_SEQ_LEN)
    
    # Optimizer and FP16 GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler('cuda')

    t_start = time.time()

    # Dynamically measure and sync step budget
    if master_process:
        print("Calibrating training throughput dynamically (with compile warmup)...")
    model.train()
    
    # Run 5 steps to warm up and trigger compilation
    for _ in range(5):
        cx, cy = train_loader.get_batch()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            clogits = model(cx)
            closs = F.cross_entropy(clogits.view(-1, clogits.size(-1)), cy.view(-1))
        scaler.scale(closs).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    
    # Measure time for next 10 steps
    t_cal_start = time.time()
    for _ in range(10):
        cx, cy = train_loader.get_batch()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            clogits = model(cx)
            closs = F.cross_entropy(clogits.view(-1, clogits.size(-1)), cy.view(-1))
        scaler.scale(closs).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    
    calibration_elapsed = time.time() - t_cal_start
    steps_per_sec = 10.0 / calibration_elapsed
    
    if master_process:
        print(f"Dynamic calibration: {steps_per_sec:.2f} steps/sec (10 steps took {calibration_elapsed:.2f}s)")
    
    step = 15
    
    # Calculate step budget on Rank 0
    if master_process:
        total_elapsed = time.time() - t_start
        max_steps = int((TIME_BUDGET - total_elapsed) * steps_per_sec) + step
    else:
        max_steps = 0

    # Broadcast max_steps and step to all ranks to guarantee identical iteration counts
    sync_tensor = torch.tensor([max_steps, step], dtype=torch.int32, device=device)
    if ddp:
        dist.broadcast(sync_tensor, src=0)
    max_steps = sync_tensor[0].item()
    step = sync_tensor[1].item()

    if master_process:
        print(f"Starting training (DDP: {ddp}, Max Steps: {max_steps})...")
    
    model.train()
    while step < max_steps:
        # Late-stage cosine decay learning rate scheduler
        warmup_steps = 100
        decay_start_step = int(0.70 * max_steps)
        if step < warmup_steps:
            current_lr = LEARNING_RATE * (step / warmup_steps)
        elif step < decay_start_step:
            current_lr = LEARNING_RATE
        else:
            # Cosine decay from LEARNING_RATE to 8e-5
            min_lr = 8e-5
            decay_ratio = (step - decay_start_step) / (max_steps - decay_start_step)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            current_lr = min_lr + coeff * (LEARNING_RATE - min_lr)
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        t0 = time.time()
        x, y = train_loader.get_batch()
        
        optimizer.zero_grad(set_to_none=True)
        
        # FP16 mixed precision forward pass
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        elapsed = time.time() - t_start
        
        if master_process and step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s | LR: {current_lr:.2e} | Scale: {scaler.get_scale()}")
            
    if master_process:
        print("Training finished. Evaluating...")
        # Unwrap model for evaluation if using DDP and compile
        raw_model = model.module if ddp else model
        eager_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
        val_loss, val_bpt, comp_ratio = evaluate_loss(eager_model, val_loader)
        
        print("\n--- RESULTS ---")
        print(f"val_loss: {val_loss:.6f}")
        print(f"val_bpt: {val_bpt:.6f}")
        print(f"comp_ratio: {comp_ratio:.6f}")
        print(f"num_params: {sum(p.numel() for p in eager_model.parameters()):,}")
        
        # Save model weights
        torch.save(eager_model.state_dict(), "model.pt")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
