import numpy as np
import os
import torch
import time
import torch.nn as nn
from model.gpt import GPT
from torch.utils.data import DataLoader
from training.dataset import TextDataset

ids = np.load("data/tokens.npy").astype(np.int64)
split = int(len(ids) * 0.9)
train_ids = ids[:split]
val_ids = ids[split:]

train_ds = TextDataset(train_ids, context_length=512, stride=512)
val_ds = TextDataset(val_ids, context_length=512, stride=512)
print("Train chunks:", len(train_ds))
print("Val chunks:  ", len(val_ds))

device = "cuda"
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

model = GPT(vocab_size=6144, emb_dim=512, num_heads=8, context_length=512, num_layers=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(model, loader, max_batches=50):
    """Average loss on up to max_batches of held-out text. No learning happens here."""
    model.eval()            # switch to evaluation mode
    total, n = 0.0, 0
    for x, y in loader:
        if n >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total += loss.item()  # turn a 1-number tensor into a plain Python number
        n += 1
    model.train()            # back to training mode
    return total / n

EPOCHS = 3
EVAL_EVERY = 250   # quiz less often, since there are ~2,660 steps now
step = 0
best_val = float("inf")
os.makedirs("checkpoints", exist_ok=True)
start = time.time()
model.train()
for epoch in range(EPOCHS):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimizer.zero_grad()          # clear old corrections
        loss.backward()                # figure out who's to blame
        optimizer.step()               # tweak the knobs
        if step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_loader)
            mins = (time.time() - start) / 60
            print(f"epoch {epoch}  step {step}  train {loss.item():.3f}  val {val_loss:.3f}  {mins:.1f} min")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "checkpoints/best.pt")
        step += 1