# Training Log

## 2026-10-01
- Set up repo and project structure.
- Chose Project Gutenberg as the pretraining corpus because it's all public domain, and there's a large number of files to choose from. Issues may present in use of archaic language (though that should make hallucinations amusing), inconsistent formatting, and removal of headers and footers. Target corpus size is TBD.
- Downloaded Frankenstein as test text.
-BPE core (text_to_bytes, get_pair_counts, merge) implemented
- Initial merge used a for loop, which reused tokens across overlapping pairs ("aaa" became 256, 256). Switched to a while loop to skip both tokens after a match. Caught it because output length didn't shrink.

## 2026-10-13

**Status:** Week 2 slipped due to work travel. Worked through week 2 design decisions and tokenizer code by phone while away from my laptop. Code below is committed but untested until I'm back at my laptop.

### Tokenizer
- Wrote the BPE training loop (`train`). It counts pairs, merges the most frequent, and records each merge in order. The recorded merges are the trained tokenizer.
- Ties between equally frequent pairs go to the first one encountered, since Python's `max()` returns the first winner and dicts keep insertion order. Not "correct" either way, but deterministic, so the same text always produces the same tokenizer.
- Two stopping conditions: stop if no pairs are left, and stop if the best pair only appears once, since merging it would just memorize one spot in the text.
- `encode` adapted from Karpathy's minbpe. Chose it over a simpler loop that replays every merge, because it only considers merges that can apply to the current text, so it scales better on a full book.
- `decode` uses `errors="replace"` because model-generated token sequences aren't guaranteed to be valid UTF-8 (multi-byte characters can get split). Without it, decoding would crash.
- `build_vocab` maps every token ID to its bytes. Order matters: a merged token can only be built once both of its parents exist.

### Design decisions
- **Vocab size: 6,144.** Divisible by 64 for GPU efficiency. Large enough to keep common words intact and sequences shorter, small enough to keep pure-Python tokenizer training reasonable and avoid rare tokens the model sees too seldom to learn. The tokenizer will be trained on a sample of the corpus rather than the full set.
- **Stride = context length.** Each chunk starts where the last ended, so no overlap. Overlapping chunks would mean far more training time spent rereading the same text, and with a smallish corpus, that pushes the model to memorize the books instead of learning how language works.
- **Context length: 512.** Long enough to capture multi-paragraph context in fiction, still manageable on 12GB. Attention cost grows faster than context length, so if memory becomes a problem, I'll reduce batch size first, then fall back to 256.
- **Embedding width: 512.** Smaller than GPT-2 small (768) to fit a modest corpus and reduce overfitting risk, while staying expressive. Divisible by 64, and splits cleanly into 8 attention heads of 64. Uses learned positional embeddings (one per slot, 0-511), added to the token embeddings.

### Next
- Test `train`, `encode`, `decode`, `build_vocab` with a round-trip check on a toy string, then Frankenstein.
- Implement the data loader and embedding layers in PyTorch.
- Begin week 3: attention.


## 2026-10-20

- Built a single attention head (model/attention.py).
- Query, key, value: Query - what self is offering, key - what self is, value - what self is worth
- Why the mask is a buffer, not learned: If learned the model oculd just change the mask to be one, fulfilling whatever it's looking for with the first thing it sees.
- Why masked scores are -inf instead of 0: 0 does not translate to 0%. -inf is always going to be the lowest there is.
- Why divide by √head_size: to "turn the volume down"
- Head size 64 = 512 / 8 heads; more heads means better training, but also more taxing training. 64 is a good nu mber for my GPU.
- transpose(-2, -1): We count form the back because we are alwasy adding things to the FRONT of our grid. The back won't change.
- Not yet run: no PyTorch installed on travel laptop setup. Shape test pending.
- Next: multi-head attention.

### Correction to 2026-10-20
- I had query and key backwards. Query is what a token is looking for. Key is what a token offers to be found by. Value is what it hands over once it's picked.

### Next
- Layer norm, feedforward block, transformer block with residuals, full model, overfit test.

## 2026-10-26

**Status:** Behind schedule (week 4, finishing week 2 and 3 work). Sent Weiss a catch-up plan; he approved async updates in place of biweekly meetings.

### Setup
- Synced repo: local attention commit and the Oct 13 web edits had diverged. Rebased, resolved a conflict in this log, pushed.
- Installed PyTorch 2.11 (CUDA 12.8) on the RTX 5070 Ti laptop. GPU detected. The 50-series needs the cu128 build or newer.

### Attention
- Single head shape test passed: (2, 4, 512) in, (2, 4, 64) out.
- Built MultiHeadAttention: 8 heads of 64 run side by side, outputs concatenated back to 512, then a final linear projection (512 to 512) that lets the heads' findings mix.
- Used nn.ModuleList instead of a plain list so PyTorch registers each head's weights for training.
- Shape test passed: (2, 4, 512) in, (2, 4, 512) out. Same shape in and out means attention layers can stack, and residual connections can add the output back onto the input.

### Layer norm
- Built LayerNorm (model/layernorm.py): subtract each token's mean, divide by its standard deviation, then apply learned scale (starts at 1) and shift (starts at 0).
- eps (1e-5) prevents dividing by zero.
- Test: input centered at 50 with spread 10 came out with mean ~0 and std ~1 per token. Passed.

### Feedforward
- Built FeedForward (model/feedforward.py): expand 512 to 2048, GELU, shrink back to 512.
- Attention lets tokens share information; feedforward lets each token process what it gathered on its own.
- GELU adds non-linearity. Without it, stacked linear layers collapse into one big linear layer.
- Shape test passed: (2, 4, 512) in, (2, 4, 512) out.

### Transformer block
- Built TransformerBlock (model/block.py): layer norm, multi-head attention, residual add; then layer norm, feedforward, residual add.
- Residuals add the input back instead of replacing it, so each block only learns what to change, and the training signal has a shortcut back through deep stacks.
- Pre-norm: layer norm goes before attention and feedforward, not after.
- Shape test passed: (2, 4, 512) in, (2, 4, 512) out.

### Embeddings
- Built Embeddings (model/embedding.py): token embedding (6,144 x 512 lookup table) plus positional embedding (512 x 512), added together.
- Token embedding learns what a token means; positional embedding learns where it sits, so word order matters.
- Shape test passed: (2, 4) token IDs in, (2, 4, 512) out. This is where IDs become the 512-wide vectors the rest of the model uses.

### Full model
- Assembled GPT (model/gpt.py): embeddings, 6 transformer blocks, final layer norm, output head (512 to 6,144, no bias).
- Shape test passed: (2, 4) token IDs in, (2, 4, 6144) out. One score per vocab token at every position; the highest score is the model's next-token guess.
- Parameter count: 25,459,712.
- Config: vocab 6,144, context 512, embedding width 512, 8 heads of 64, 6 layers.
- Layers set to 6 for now: faster feedback during the overfit test and trial runs. Will revisit (possibly 12, ~44M params) in week 5 once corpus size is set, since larger models need more data to avoid memorizing it.

## 2026-10-27

### Tokenizer
- encode and decode were missing from bpe.py. They were written by phone on 10-13 but never made it into the file. Rewrote both.
- decode: look up each ID's bytes in the vocab, join them, decode as UTF-8 with errors="replace".
- encode: start from raw bytes, repeatedly merge the pair that was learned earliest (lowest merge ID), stop when no pair has a merge rule. Earliest first, because later merges are built from earlier ones.
- Toy test passed: "aaabdaaabac" (11 bytes) encodes to 5 tokens and decodes back to the original exactly.

- Frankenstein round trip passed: trained 256 merges on the first 50,000 characters, then encoded and decoded the whole book (441,144 bytes). Output matched exactly, including text the tokenizer never trained on.
- 441,144 bytes became 206,066 tokens (about 2.1 bytes per token) with only 256 merges. Expect closer to 4 bytes per token with the full 6,144 vocab.
- First merges learned: "e ", "th", "d ", "er", "s ", "in". These are the most common pairs in English, plus end-of-word patterns, found purely by counting.
- Speed: training on 50k characters takes about 1 second; encoding the whole book takes about 11 seconds. Pure Python is fine for now.