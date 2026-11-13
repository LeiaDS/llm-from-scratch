import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.gpt import GPT
from training.dataset import TextDataset

device = "cuda"
ids = np.load("data/tokens.npy").astype(np.int64)
split = int(len(ids) * 0.9)
val_ids = ids[split:]                    # just the held-out 10%
val_loader = DataLoader(TextDataset(val_ids, 512, 512), batch_size=8)

LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
model = GPT(vocab_size=6144, emb_dim=512, num_heads=8, context_length=512, num_layers=LAYERS).to(device)
model.load_state_dict(torch.load(sys.argv[1]))   # checkpoint path from the command line
model.eval()

loss_fn = nn.CrossEntropyLoss()
total, n = 0.0, 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()
        n += 1
print(f"{sys.argv[1]}: val loss {total / n:.3f} over {n} batches")