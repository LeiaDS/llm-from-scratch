import json
import torch
from model.gpt import GPT
from tokenizer.bpe import encode, decode, build_vocab

device = "cuda"
with open("tokenizer/merges.json") as f:
    merges = {(a, b): idx for a, b, idx in json.load(f)}
vocab = build_vocab(merges)

model = GPT(vocab_size=6144, emb_dim=512, num_heads=8, context_length=512, num_layers=6).to(device)
model.load_state_dict(torch.load("checkpoints/best.pt"))
model.eval()          # quiz mode, not training mode

@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.8):
    ids = torch.tensor([encode(prompt, merges)], device=device)   # shape: (1, T)
    for _ in range(max_new_tokens):
        context = ids[:, -512:]              # model can't see more than 512 tokens
        logits = model(context)              # shape: ???
        logits = logits[:, -1, :]        # keep only ONE position's scores
        probs = torch.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)        # pick one, shape (1, 1)
        ids = torch.cat([ids, next_id], dim=1)              # glue it on the end
    return decode(ids[0].tolist(), vocab)

print(generate("It was a dark and stormy night"))