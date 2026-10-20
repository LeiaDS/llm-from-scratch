import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, emb_dim, head_size, context_length):
        super().__init__()
        # Step 1: make the three projections
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.key =   nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)

        # Step 3 prep: the mask (we'll do this later)
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        T = x.shape[1]
        #x is (B, T, emb_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1)
        scores /= k.shape[-1] ** 0.5

        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        return out

if __name__ == "__main__":
    head = Head(emb_dim=512, head_size=64, context_length=512)
    x = torch.randn(2, 4, 512)   # 2 sentences, 4 tokens each, 512 numbers per token
    out = head(x)
    print(out.shape)