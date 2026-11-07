import os
from tokenizer.bpe import split_text, count_chunks

sample = []
for name in sorted(os.listdir("data/gutenberg")):
    with open(os.path.join("data/gutenberg", name), encoding="utf-8") as f:
        text = f.read()
    middle = len(text) // 2
    sample.append(text[middle : middle + 33_000])
sample = "\n".join(sample)

chunks = split_text(sample)
counts = count_chunks(sample)
print("Total chunks: ", len(chunks))
print("Unique chunks:", len(counts))
print("Top 5:", sorted(counts.items(), key=lambda kv: -kv[1])[:5])