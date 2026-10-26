import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)

    def forward(self, idx):
        T = idx.shape[1]
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        return tok + pos

if __name__ == "__main__":
    emb = Embeddings(vocab_size=6144, emb_dim=512, context_length=512)
    idx = torch.randint(0, 6144, (2, 4))   # 2 sentences, 4 fake token IDs each
    print(idx.shape)
    print(emb(idx).shape)