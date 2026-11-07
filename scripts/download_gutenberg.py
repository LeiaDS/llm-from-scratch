import os
import re
import time
import urllib.request
import truststore
truststore.inject_into_ssl()

# Project Gutenberg IDs: 19th-century novels originally written in English
BOOKS = [
    # Austen
    1342, 161, 158, 105, 121, 141,
    # Shelley, Stoker, Le Fanu, Stevenson, Wilde (Gothic)
    84, 345, 10007, 43, 174,
    # The Brontës
    1260, 768, 767, 969,
    # Dickens
    98, 1400, 46, 730, 766, 1023, 580, 786,
    # Eliot, Gaskell, Thackeray, Collins
    145, 6688, 550, 4276, 394, 599, 155, 583,
    # Hardy
    110, 107, 143,
    # Doyle
    244, 2097, 1661, 834,
    # Wells, Stevenson, Kipling, Conrad, James
    35, 36, 5230, 159, 120, 236, 219, 209,
    # Carroll, Sewell, Scott
    11, 12, 271, 82,
    # Americans: Melville, Hawthorne, Twain, Alcott, Poe, Irving, Chopin, Thoreau
    2701, 25344, 76, 74, 1837, 514, 2147, 2148, 41, 160, 205,
]

OUT_DIR = "data/gutenberg"
os.makedirs(OUT_DIR, exist_ok=True)


def strip_boilerplate(text):
    """Keep only the book itself, dropping Gutenberg's license header and footer."""
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start() - (start.end() if start else 0)]
    return text.strip()


for book_id in BOOKS:
    path = os.path.join(OUT_DIR, f"{book_id}.txt")
    if os.path.exists(path):
        print(f"{book_id}: already have it, skipping")
        continue
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"{book_id}: FAILED ({e})")
        continue
    title = re.search(r"Title: (.*)", raw)
    title = title.group(1).strip() if title else "unknown title"
    text = strip_boilerplate(raw)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{book_id}: {title} ({len(text):,} chars)")
    time.sleep(2)  # be polite to Gutenberg's servers

total = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
print(f"\nDone. {len(os.listdir(OUT_DIR))} books, {total / 1_000_000:.1f} MB")