import json
from tokenizer.bpe import build_vocab

with open("tokenizer/merges.json") as f:
    merges = {(a, b): idx for a, b, idx in json.load(f)}
vocab = build_vocab(merges)

for idx in range(6124, 6144):
    print(idx, repr(vocab[idx].decode("utf-8", errors="replace")))