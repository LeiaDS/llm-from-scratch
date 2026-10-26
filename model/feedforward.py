import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    ff = FeedForward(512)
    x = torch.randn(2, 4, 512)
    print(ff(x).shape)