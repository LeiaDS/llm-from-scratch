import json, os, time
import numpy as np
from tokenizer.bpe import encode

BOOK_DIR = "data/gutenberg"
OUT = "data/tokens.npy"

with open("tokenizer/merges.json") as f:
    merges = {(a, b): idx for a, b, idx in json.load(f)}

cache = {}      # shared across ALL books, so " the" is solved once total
all_ids = []
start = time.time()
for name in sorted(os.listdir(BOOK_DIR)):
    with open(os.path.join(BOOK_DIR, name), encoding="utf-8") as f:
        text = f.read()
    all_ids.extend(encode(text, merges, cache))

ids = np.array(all_ids, dtype=np.uint16)
np.save(OUT, ids)
print(f"{len(ids):,} tokens, {len(cache):,} unique chunks, {time.time() - start:.1f}s")