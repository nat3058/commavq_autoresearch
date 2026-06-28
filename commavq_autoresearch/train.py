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
N_LAYER = 8
N_HEAD = 7
N_EMBD = 448

TOKEN_EMBD_DIM = 64
BATCH_SIZE = 64          # Batch size per GPU (effective batch size = 128)
LEARNING_RATE = 8e-4     # Restored to optimal learning rate
WEIGHT_DECAY = 0.01


# ---------------------------------------------------------------------------
# GPT Model Components
# ---------------------------------------------------------------------------
class FrameEmbedding(nn.Module):
    def __init__(self, vocab_size, token_embd_dim, n_embd, frame_dim=FRAME_DIM):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, token_embd_dim)
        self.conv = nn.Conv2d(token_embd_dim, token_embd_dim, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(8, token_embd_dim)
        self.proj = nn.Linear(frame_dim * token_embd_dim, n_embd, bias=False)
        
    def forward(self, x):
        B, L, frame_dim = x.size()
        emb = self.wte(x) # (B, L, 128, token_embd_dim)
        # Reshape to 2D spatial grid (B * L, token_embd_dim, 8, 16)
        emb_grid = emb.view(B * L, 8, 16, emb.size(-1)).permute(0, 3, 1, 2)
        emb_grid = emb_grid + F.silu(self.gn(self.conv(emb_grid)))
        # Reshape back to flat sequence
        emb = emb_grid.permute(0, 2, 3, 1).contiguous().view(B, L, -1)
        return self.proj(emb) # (B, L, n_embd)


class FrameHead(nn.Module):
    def __init__(self, n_embd, token_embd_dim, frame_dim=FRAME_DIM):
        super().__init__()
        self.proj = nn.Linear(n_embd, frame_dim * token_embd_dim, bias=False)
        self.conv1 = nn.Conv2d(token_embd_dim, token_embd_dim, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, token_embd_dim)
        self.conv2 = nn.Conv2d(token_embd_dim, token_embd_dim, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, token_embd_dim)
        self.conv3 = nn.Conv2d(token_embd_dim, token_embd_dim, kernel_size=3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(8, token_embd_dim)
        self.frame_dim = frame_dim
        self.token_embd_dim = token_embd_dim
        
    def forward(self, x, wte_weight):
        B, L, C = x.size()
        features = self.proj(x) # (B, L, 128 * token_embd_dim)
        # Reshape to 2D spatial grid (B * L, token_embd_dim, 8, 16)
        features = features.view(B * L, 8, 16, self.token_embd_dim).permute(0, 3, 1, 2)
        
        # Spatial coordination refinement (3 residual layers)
        features = features + F.silu(self.gn1(self.conv1(features)))
        features = features + F.silu(self.gn2(self.conv2(features)))
        features = features + F.silu(self.gn3(self.conv3(features)))
        
        # Reshape back to flat tokens
        features = features.permute(0, 2, 3, 1).contiguous().view(B, L, self.frame_dim, self.token_embd_dim)
        logits = torch.matmul(features, wte_weight.T) # (B, L, 128, 1024)
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
        
        # Construct Multi-Scale Causal Mask of shape (1, n_head, T, T)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1) # (T, T)
        mask = mask.unsqueeze(0).unsqueeze(1).repeat(1, self.n_head, 1, 1) # (1, n_head, T, T)
        
        # Local causal heads (Head 3 and 4: window size 4)
        local_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        local_lower = torch.tril(torch.full((T, T), float('-inf'), device=x.device), diagonal=-4)
        local_mask = local_mask + local_lower
        mask[:, 3:5, :, :] = local_mask
        
        # Strided causal heads (Head 5 and 6: every 2nd frame)
        t_indices = torch.arange(T, device=x.device).unsqueeze(1)
        j_indices = torch.arange(T, device=x.device).unsqueeze(0)
        stride_mismatch = ((t_indices - j_indices) % 2 != 0)
        strided_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        strided_mask[stride_mismatch] = float('-inf')
        mask[:, 5:7, :, :] = strided_mask
        
        # Run attention with the custom mask
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
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
    
    # Initialize model
    model = GPT(VOCAB_SIZE, TOKEN_EMBD_DIM, N_EMBD, N_HEAD, N_LAYER, MAX_SEQ_LEN).to(device)
    model = torch.compile(model, mode="reduce-overhead")
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Setup data loaders
    train_loader = Dataloader(TRAIN_BIN, BATCH_SIZE, MAX_SEQ_LEN)
    val_loader = Dataloader(VAL_BIN, BATCH_SIZE, MAX_SEQ_LEN)
    
    # Optimizer and FP16 GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler('cuda')

    
    if master_process:
        print(f"Starting training (DDP: {ddp}, Time Budget: {TIME_BUDGET}s)...")
    t_start = time.time()
    step = 0
    
    model.train()
    while True:
        t0 = time.time()
        x, y = train_loader.get_batch()
        
        optimizer.zero_grad()
        
        # FP16 mixed precision forward pass
        with torch.amp.autocast('cuda', dtype=torch.float16):
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
        
        verify_arithmetic_coding(raw_model, comp_ratio)

    if ddp:
        dist.destroy_process_group()

def verify_arithmetic_coding(model, comp_ratio):
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    print("\nInstalling torchac to verify arithmetic coding...")
    os.system("pip install torchac > /dev/null 2>&1")
    try:
        import torchac
    except ImportError:
        print("Verification skipped: torchac not installed.")
        return
        
    model.eval()
    from prepare import VAL_BIN, FRAME_DIM
    if not os.path.exists(VAL_BIN):
        print("Verification skipped: VAL_BIN not found.")
        return
        
    val_data = np.fromfile(VAL_BIN, dtype=np.int16).reshape(-1, FRAME_DIM)
    if len(val_data) < 1200:
        print("Verification skipped: val data too small.")
        return
        
    # Take first 1200 frames (1 segment)
    gt_tokens = torch.from_numpy(val_data[:1200].astype(np.int64)).cuda() # (1200, 128)
    compressed_bytes_list = []
    
    # Compress frame 0 using a uniform prior
    flat_probs = torch.full((128, 1024), 1.0 / 1024.0, device='cuda')
    flat_cdf = torch.zeros(128, 1025, device='cuda')
    flat_cdf[:, 1:] = torch.cumsum(flat_probs, dim=-1)
    flat_cdf = flat_cdf / flat_cdf[:, -1:]
    
    byte_stream = torchac.encode_float_cdf(flat_cdf.cpu(), gt_tokens[0].to(torch.int16).cpu())
    compressed_bytes_list.append(byte_stream)
    
    # Compress frames 1 to 1199
    history = gt_tokens.unsqueeze(0) # (1, 1200, 128)
    print("Running autoregressive compression...")
    for t in range(1, 1200):
        start_idx = max(0, t - 32)
        context = history[:, start_idx:t, :] # (1, context_len, 128)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(context)
        
        target_logits = logits[0, -1, :, :].float() # (128, 1024)
        probs = torch.softmax(target_logits, dim=-1)
        
        cdf = torch.zeros(128, 1025, device='cuda')
        cdf[:, 1:] = torch.cumsum(probs, dim=-1)
        cdf = cdf / cdf[:, -1:]
        
        target_symbols = gt_tokens[t].to(torch.int16)
        byte_stream = torchac.encode_float_cdf(cdf.cpu(), target_symbols.cpu())
        compressed_bytes_list.append(byte_stream)
        
    total_encoded_bytes = sum(len(b) for b in compressed_bytes_list)
    uncompressed_bytes = 1200 * 128 * 10 / 8
    actual_ratio = uncompressed_bytes / total_encoded_bytes
    
    # Autoregressive Decompression
    print("Running autoregressive decompression...")
    decoded_history = torch.zeros((1, 1200, 128), dtype=torch.int64, device='cuda')
    
    # Decode frame 0
    decoded_symbols_0 = torchac.decode_float_cdf(flat_cdf.cpu(), compressed_bytes_list[0])
    decoded_history[0, 0, :] = torch.from_numpy(np.array(decoded_symbols_0)).cuda()
    
    # Decode frames 1 to 1199
    for t in range(1, 1200):
        start_idx = max(0, t - 32)
        context = decoded_history[:, start_idx:t, :]
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(context)
                
        target_logits = logits[0, -1, :, :].float()
        probs = torch.softmax(target_logits, dim=-1)
        
        cdf = torch.zeros(128, 1025, device='cuda')
        cdf[:, 1:] = torch.cumsum(probs, dim=-1)
        cdf = cdf / cdf[:, -1:]
        
        decoded_symbols = torchac.decode_float_cdf(cdf.cpu(), compressed_bytes_list[t])
        decoded_history[0, t, :] = torch.from_numpy(np.array(decoded_symbols)).cuda()
        
    assert torch.equal(gt_tokens, decoded_history[0]), "Lossless decompression failed! Decoded tokens differ from original."
    print("SUCCESS: Lossless decompression verified (100% identical).")
    print(f"Actual compressed size (1 segment): {total_encoded_bytes:,} bytes.")
    print(f"Uncompressed size (1 segment): {int(uncompressed_bytes):,} bytes.")
    print(f"Actual compression ratio: {actual_ratio:.6f}x (vs theoretical {comp_ratio:.6f}x)")

    # Save model weights
    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    train()

