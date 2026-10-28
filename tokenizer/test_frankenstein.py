import glob
import time
from tokenizer.bpe import train, build_vocab, encode, decode

path = glob.glob("data/*.txt")[0]
print("Book:", path)

with open(path, encoding="utf-8") as f:
    text = f.read()

# Train on a small sample with a small vocab, just to test. Real vocab (6,144) comes later.
start = time.time()
merges = train(text[:50000], 512)
vocab = build_vocab(merges)
print("Trained", len(merges), "merges in", round(time.time() - start, 1), "sec")

# Round trip on the WHOLE book
start = time.time()
ids = encode(text, merges)
print("Encoded in", round(time.time() - start, 1), "sec")

print("Bytes in:", len(text.encode("utf-8")))
print("Tokens out:", len(ids))
print("Round trip:", decode(ids, vocab) == text)

# Peek at a few learned tokens
for idx in range(256, 266):
    print(idx, vocab[idx])