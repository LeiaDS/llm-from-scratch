import csv
import random
import re

CATALOG = "data/catalog/pg_catalog.csv"
BIRTH_MIN, BIRTH_MAX = 1770, 1870
TARGET = 500

def birth_year(authors):
    """First author's birth year, like 1743 from 'Jefferson, Thomas, 1743-1826'. None if missing."""
    m = re.search(r"(\d{4})-", authors)
    return int(m.group(1)) if m else None

picks = []
with open(CATALOG, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["Type"] != "Text":
            continue
        if row["Language"] != "en":
            continue
        if "Fiction" not in row["Subjects"]:
            continue
        if not row["LoCC"].startswith(("PR", "PS")):
            continue
        year = birth_year(row["Authors"])
        if year is None or not (BIRTH_MIN <= year <= BIRTH_MAX):
            continue
        picks.append(int(row["Text#"]))

print("Matching books:", len(picks))
random.seed(42)
chosen = random.sample(picks, min(TARGET, len(picks)))
with open("data/catalog/book_ids.txt", "w") as f:
    f.write("\n".join(str(i) for i in sorted(chosen)))
print("Saved", len(chosen), "IDs")