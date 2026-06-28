import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 32          # Context length in frames (e.g. 32 frames)
TIME_BUDGET = 300         # 5 minutes training time budget
EVAL_BATCHES = 50         # Validation batches
VOCAB_SIZE = 1024         # 1024 VQ tokens (0-1023), no special tokens needed
FRAME_DIM = 128           # 128 VQ tokens per frame

CACHE_DIR = "/kaggle/tmp/data_cache"
TRAIN_BIN = os.path.join(CACHE_DIR, "train.bin")
VAL_BIN = os.path.join(CACHE_DIR, "val.bin")

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data():
    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        print(f"Data already prepared at {CACHE_DIR}")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Downloading dataset from Hugging Face...")
    
    # Load first split for training
    train_ds = load_dataset('commaai/commavq', data_files={'train': ['data-0000.tar.gz']})['train']
    # Load second split for validation, take a subset of 100 segments
    val_ds = load_dataset('commaai/commavq', data_files={'train': ['data-0001.tar.gz']})['train'].select(range(100))
    
    print("Processing training split...")
    train_tokens = []
    for x in train_ds:
        tok = np.array(x['token.npy'], dtype=np.int16).reshape(-1)
        train_tokens.append(tok)
    train_arr = np.concatenate(train_tokens)
    
    print("Processing validation split...")
    val_tokens = []
    for x in val_ds:
        tok = np.array(x['token.npy'], dtype=np.int16).reshape(-1)
        val_tokens.append(tok)
    val_arr = np.concatenate(val_tokens)
    
    # Save to binary files
    train_arr.tofile(TRAIN_BIN)
    val_arr.tofile(VAL_BIN)
    print(f"Saved prepared binary files to {CACHE_DIR}")

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------
class Dataloader:
    def __init__(self, filename, batch_size, sequence_len):
        # Load entire binary file to RAM and convert to int64 GPU tensor
        raw_data = np.fromfile(filename, dtype=np.int16)
        self.frames = torch.from_numpy(raw_data.reshape(-1, FRAME_DIM).astype(np.int64)).cuda()
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.num_frames = len(self.frames)
        
    def get_batch(self):
        # Generate random start indices directly on GPU
        ix = torch.randint(0, self.num_frames - self.sequence_len - 1, (self.batch_size,), device='cuda')
        x = torch.stack([self.frames[i : i + self.sequence_len] for i in ix])
        y = torch.stack([self.frames[i + 1 : i + 1 + self.sequence_len] for i in ix])
        return x, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model, val_loader):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = val_loader.get_batch()
        logits = model(x)
        # logits shape: (Batch, L, 128, 1024), y shape: (Batch, L, 128)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        losses.append(loss.item())
    model.train()
    val_loss = np.mean(losses)
    val_bpt = val_loss / math.log(2)
    comp_ratio = 10.0 / val_bpt
    return val_loss, val_bpt, comp_ratio

if __name__ == "__main__":
    prepare_data()
