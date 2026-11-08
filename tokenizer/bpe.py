import regex as re
from collections import Counter

SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def normalize(text):
    """Make text consistent before splitting."""
    return text.replace("\u2019", "'").replace("\u2018", "'")

def split_text(text):
    """Chop text into chunks. Merges will never cross a chunk boundary."""
    return re.findall(SPLIT_PATTERN, normalize(text))

def text_to_bytes(text):
    """Convert a string into a list of integers 0-255 (UTF-8 bytes)."""
    return list(text.encode("utf-8"))

def count_chunks(text):
    """Split text and count each unique chunk. Return {tuple_of_byte_ids: count}."""
    chunk_counts = Counter(split_text(text))
    return {tuple(text_to_bytes(chunk)): count for chunk, count in chunk_counts.items()}

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

def train(text, vocab_size):
    words = count_chunks(text)          # {tuple_of_ids: count}
    merges = {}
    for k in range(vocab_size - 256):
        # Count pairs inside every unique chunk, weighted by how often that chunk appears
        counts = {}
        for word, freq in words.items():
            for pair, c in get_pair_counts(list(word)).items():
                counts[pair] = counts.get(pair, 0) + c * freq
        if not counts:
            break
        pair = max(counts, key=counts.get)
        if counts[pair] < 2:
            break
        new_id = 256 + k
        # Apply the merge inside every chunk
        words = {tuple(merge(list(word), pair, new_id)): freq for word, freq in words.items()}
        merges[pair] = new_id
    return merges

def build_vocab(merges):
    """Map every token ID to the bytes it represents."""
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

def decode(ids, vocab):
    """Turn a list of token IDs back into text."""
    text_bytes = b"".join(vocab[idx] for idx in ids)
    return text_bytes.decode("utf-8", errors="replace")

def encode_chunk(text, merges):
    """Turn text into a list of token IDs using the learned merges."""
    ids = text_to_bytes(text)
    while len(ids) >= 2:
        counts = get_pair_counts(ids)
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def encode(text, merges, cache=None):
    """Turn text into token IDs one chunk at a time, reusing past results."""
    if cache is None:
        cache = {}
    ids = []
    for chunk in split_text(text):
        if chunk not in cache:
            cache[chunk] = encode_chunk(chunk, merges)
        ids.extend(cache[chunk])
    return ids

if __name__ == "__main__":
    text = "aaabdaaabac"
    merges = train(text, 259)
    vocab = build_vocab(merges)
    ids = encode(text, merges)
    print(merges)
    print(ids)
    print(decode(ids, vocab) == text)
