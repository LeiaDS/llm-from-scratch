import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, ids, context_length, stride):
        self.inputs = []
        self.targets = []
        for i in range(0, len(ids) - context_length, stride):
            chunk = ids[i : i + context_length + 1]
            self.inputs.append(torch.tensor(chunk[:-1]))
            self.targets.append(torch.tensor(chunk[1:]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]

if __name__ == "__main__":
    ids = list(range(20))   # fake tokens: 0, 1, 2 ... 19
    ds = TextDataset(ids, context_length=4, stride=4)
    print(len(ds))
    for x, y in ds:
        print(x.tolist(), "->", y.tolist())

    # Real test: Frankenstein
    from torch.utils.data import DataLoader
    from tokenizer.bpe import train, encode

    with open("data/frankenstein.txt", encoding="utf-8") as f:
        text = f.read()

    merges = train(text[:50000], 512)
    ids = encode(text, merges)

    ds = TextDataset(ids, context_length=512, stride=512)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    print("Chunks:", len(ds))
    print("Batches:", len(loader))

    x, y = next(iter(loader))
    print("Input batch:", x.shape)
    print("Target batch:", y.shape)