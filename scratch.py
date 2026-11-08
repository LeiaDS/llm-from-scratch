import json, time
from tokenizer.bpe import encode, decode, build_vocab, normalize

with open("tokenizer/merges.json") as f:
    merges = {(a, b): idx for a, b, idx in json.load(f)}
vocab = build_vocab(merges)

with open("data/frankenstein.txt", encoding="utf-8") as f:
    text = f.read()

start = time.time()
ids = encode(text, merges)
print(f"{len(text):,} chars -> {len(ids):,} tokens in {time.time() - start:.1f}s")
print("Round trip:", decode(ids, vocab) == normalize(text))