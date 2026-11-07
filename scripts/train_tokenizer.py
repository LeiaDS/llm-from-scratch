import json
import os
import time
from tokenizer.bpe import train

BOOK_DIR = "data/gutenberg"
SAMPLE_PER_BOOK = 33_000   # characters from each book, about 2 MB total
VOCAB_SIZE = 6144

# Take a slice from the middle of every book, skipping title pages and front matter
sample = []
for name in sorted(os.listdir(BOOK_DIR)):
    with open(os.path.join(BOOK_DIR, name), encoding="utf-8") as f:
        text = f.read()
    middle = len(text) // 2
    sample.append(text[middle : middle + SAMPLE_PER_BOOK])
sample = "\n".join(sample)
print(f"Sample: {len(sample):,} characters from {len(os.listdir(BOOK_DIR))} books")

start = time.time()
merges = train(sample, VOCAB_SIZE)
print(f"Learned {len(merges):,} merges in {(time.time() - start) / 60:.1f} minutes")

# JSON can't store tuple keys, so save each merge as [first, second, new_id]
with open("tokenizer/merges.json", "w") as f:
    json.dump([[p0, p1, idx] for (p0, p1), idx in merges.items()], f)
print("Saved to tokenizer/merges.json")