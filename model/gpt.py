import torch
import torch.nn as nn
from model.embedding import Embeddings
from model.block import TransformerBlock
from model.layernorm import LayerNorm

class GPT(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, context_length, num_layers):
        super().__init__()
        self.emb = Embeddings(vocab_size, emb_dim, context_length)
        self.blocks = nn.Sequential(*[TransformerBlock(emb_dim, num_heads, context_length) for _ in range(num_layers)])
        self.ln_f = LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

if __name__ == "__main__":
    model = GPT(vocab_size=6144, emb_dim=512, num_heads=8, context_length=512, num_layers=6)
    idx = torch.randint(0, 6144, (2, 4))
    print(model(idx).shape)
    print(sum(p.numel() for p in model.parameters()))