def text_to_bytes(text):
    """Convert a string into a list of integers 0-255 (UTF-8 bytes)."""
    return list(text.encode("utf-8"))


def get_pair_counts(ids):
    """Count every adjacent pair in ids. Return a dict like {(97, 97): 4, ...}."""
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Return a new list where every occurrence of pair is replaced by new_id."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


if __name__ == "__main__":
    ids = text_to_bytes("aaabdaaabac")
    print(ids)
    print(get_pair_counts(ids))
    print(merge(ids, (97, 97), 256))