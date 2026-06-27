import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, TRAIN_BIN, VAL_BIN, Dataloader, evaluate_loss

# ---------------------------------------------------------------------------
# Hyperparameters (agent modifies these)
# ---------------------------------------------------------------------------
N_LAYER = 4
N_HEAD = 8
N_EMBD = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

# ---------------------------------------------------------------------------
# GPT Model Components
# ---------------------------------------------------------------------------
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

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx):
        device = idx.device
        t = idx.size(1)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ---------------------------------------------------------------------------
# Training Execution
# ---------------------------------------------------------------------------
def train():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    
    # Initialize model
    model = GPT(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER, MAX_SEQ_LEN).cuda()
    model = torch.compile(model)
    
    # Setup data loaders
    train_loader = Dataloader(TRAIN_BIN, BATCH_SIZE, MAX_SEQ_LEN)
    val_loader = Dataloader(VAL_BIN, BATCH_SIZE, MAX_SEQ_LEN)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)
    
    print(f"Starting training (Time Budget: {TIME_BUDGET}s)...")
    t_start = time.time()
    step = 0
    
    model.train()
    while True:
        t0 = time.time()
        x, y = train_loader.get_batch()
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        elapsed = time.time() - t_start
        
        if step % 20 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
            
        if elapsed >= TIME_BUDGET:
            break
            
    print("Training finished. Evaluating...")
    val_loss, val_bpt, comp_ratio = evaluate_loss(model, val_loader)
    
    print("\n--- RESULTS ---")
    print(f"val_loss: {val_loss:.6f}")
    print(f"val_bpt: {val_bpt:.6f}")
    print(f"comp_ratio: {comp_ratio:.6f}")
    print(f"num_params: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    train()
