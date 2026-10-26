import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.layernorm import LayerNorm
from model.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, context_length):
        super().__init__()
        self.ln1 = LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads, context_length)
        self.ln2 = LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

if __name__ == "__main__":
    block = TransformerBlock(emb_dim=512, num_heads=8, context_length=512)
    x = torch.randn(2, 4, 512)
    print(block(x).shape)