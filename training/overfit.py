import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.gpt import GPT
from tokenizer.bpe import train, encode
from training.dataset import TextDataset

device = "cuda"

# Get ONE real batch of Frankenstein
with open("data/frankenstein.txt", encoding="utf-8") as f:
    text = f.read()
merges = train(text[:50000], 512)
ids = encode(text, merges)
ds = TextDataset(ids, context_length=512, stride=512)
x, y = next(iter(DataLoader(ds, batch_size=8)))
x, y = x.to(device), y.to(device)

model = GPT(vocab_size=6144, emb_dim=512, num_heads=8, context_length=512, num_layers=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

for step in range(200):
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(step, round(loss.item(), 4))

print("Final loss:", round(loss.item(), 4))