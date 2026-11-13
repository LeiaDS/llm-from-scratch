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

### Data loader
- Built TextDataset (training/dataset.py): slides a window across the token IDs, grabbing context_length + 1 tokens per chunk. Input is the chunk minus its last token, target is the chunk minus its first, so every position's target is the next token.
- Stride = context length, so chunks don't overlap (per the week 2 decision). Leftover tokens at the end that can't fill a full chunk are dropped.
- DataLoader stacks chunks into shuffled batches of 8.
- Frankenstein test: about 206,000 tokens became 402 chunks of 512, in 51 batches (the last one has 2 chunks). Input and target batches are both (8, 512), matching what the model expects.
- This closes out week 2.

### Overfit test
- Built training/overfit.py: one real batch of Frankenstein (8 x 512 tokens), trained for 200 steps with AdamW (lr 3e-4) and cross-entropy loss, on the RTX 5070 Ti.
- Loss went from 8.91 to 0.0037. The starting loss matches random guessing across 6,144 tokens (about 8.7), and the model memorized the batch almost perfectly.
- This confirms the model, loss, backpropagation, and optimizer all work together. Cleared for pretraining.
- The training loop: forward pass, compute loss, zero_grad (clear old gradients), backward (work out how to nudge each weight), step (apply the nudges).
- Run took about 2 minutes, mostly pure-Python tokenization. Plan for week 5: tokenize the corpus once and save the token IDs to disk.
- Weeks 2 and 3 are now complete.
- Base rundown of the loop:
-logits: the model’s raw scores for what comes next
-loss: compares the guess to the real answer, boils it down to -one wrongness number
-zero_grad: clears out the old corrections so they don’t pile up
-backward: figures out who’s to blame and which way to nudge each knob
-step: tweaks the knobs based on who’s at fault

## 2026-11-06
### Corpus
- Chose a 19th-century corpus: 61 novels originally written in English, roughly 1810 to 1900 (Austen, the Brontës, Dickens, Eliot, Hardy, Gothic horror, Doyle, Wells, and American authors). 43 MB total, an estimated 10 to 11 million tokens.
- Kept to one era so the model learns one consistent style instead of a blend.
- scripts/download_gutenberg.py downloads each book and strips Project Gutenberg's license header and footer, so the model doesn't learn legal boilerplate.
- Tradeoff: 25.5M parameters would ideally want around 500M tokens (about 20 per parameter). This corpus is much smaller, so some memorization is expected. Validation loss will show how much.
- Fixed an SSL certificate error on Windows with truststore (uses the Windows certificate store) instead of disabling verification.

### Tokenizer training
- Trained the full tokenizer (6,144 vocab, 5,888 merges) on a 2 MB sample: 33,000 characters from the middle of each of the 61 books, so every author shapes the vocabulary and title pages are skipped.
- Took 14 minutes in pure Python. Training on the full 43 MB would take hours, and a sample is enough to learn the common patterns.
- Saved merges to tokenizer/merges.json so the tokenizer can be reused without retraining.
- Next problem: encode is too slow to run on the full corpus with this many merges. Needs a faster approach before the corpus can be tokenized.

## 2026-11-07
### Pre-splitting
- Problem: the first tokenizer learned tokens that crossed word boundaries, like 'at once', '\n\nhad', and ') "'. BPE only counts byte pairs, so it merges whatever is most common.
- Fix: split text into chunks with GPT-2's regex pattern before BPE. Merges can't cross chunks, so tokens won't be glued together across multiple words or punctuation.
- The space attaches to the front of the next word (" married") because space + word is far fewer vocab slots than word + punctuation. What comes before a word is almost always a space, what comes after is chaos incarnate. Punctuation stays separate because there's many different kinds, and they're less predictable.
- Kept the space glued on instead of making it its own token because it eats up too many of my 512 context slots.
- Prediction check: "’Tis a pity she’s gone." splits into 8 chunks: ', Tis, a, pity, she, 's, gone, .. The "'" before "Tis" stays alone because 'T isn't a contraction, but 't is because the rules are case sensitive.

### Apostrophes
- Corpus has 118,845 curly ’ and 6,166 straight '. The pattern's contraction rules only match straight '.
- Normalized both ’ and ‘ to ' inside the tokenizer (not the download script) because input at inference time coming from different mediums (phone, computer) will be different.
- Tradeoff: model can't tell between opening single quotes and closing ones. It has to use context. The 6,166 straight ' already worked this way.

### Training
- Updated training to count pairs over unique chunks weighted by frequency. Of 493,163 chunks, only 27,083 were unique. Cut time from 14 minutes to 2 minutes. I predicted 6. First list of tokens:
```
6126 'nine '
6127 ') “'
6128 'at once'
6129 'Lac'
6130 'trying to '
6131 'beautiful '
6132 '\n\nhad '
6133 'She was '
6134 'live'
6135 'opportunity '
6136 'non'
6137 'squ'
6138 'written '
6139 'led\n\n'
6140 'tual '
6141 'pare '
6142 'Look'
6143 'within'
```
Second test created the following:
```
6124 ' endeavoured'
6125 'forth'
6126 'prevail'
6127 ' Gl'
6128 ' friendship'
6129 'det'
6130 ' proposed'
6131 'rounded'
6132 ' compassion'
6133 'refle'
6134 ' solitude'
6135 'fection'
6136 ' cousins'
6137 ' awake'
6138 ' curtains'
6139 ' pressed'
6140 ' pupils'
6141 ' dislike'
6142 ' wept'
6143 'bes'
```

-List one was random looking phrases and chunks, while list two provided (mostly) full words. Word portions (like 'refle') are portions of rarer words.

### Corpus Tokenization

-After caching encoding, full corpus tokenization came in around 5 seconds, 12.1M tokens, 45MB.

### Generation
- Ran three epochs
- Added generate.py and tested. first words generated-
"prevented a stroll, and I have only to thank you it; for if I lose, Mr. Dixon, I should have half so much time to write, I don't know, I can't wish at all about it, and say"

- 6 epochs:
- Epochs 3-5 essentially plateaued. Val bottomed out at around 3.7. Sticking to six layers instead of twelve, val-training gap indicate memorization, not learning.
- Condensed token size to 24 MB, using six layers instead of 12. 12 would simply promote more memorization, not actual learning. Amount of data is the bottleneck, not the processing.
- Best val was 3.721 at step 12,750.


 ## 2026-11-08
### Corpus expansion
- Why: the 61-book model plateaued around epoch 4, so a longer run would only memorize. More data is what earns more training.
- Wrote scripts/pick_books.py to filter Gutenberg's catalog: English text, "Fiction" in subjects, LoCC PR or PS, first author born 1770 to 1870. PR/PS (English and American literature shelves) keeps out translations because English first texts will sound different and provide a more locked-in vocab.
- 7,722 books matched. Sampled 500 with random.seed(42) so the same random 500 books are used if I lose everything in a horrible explosion.
- Merged with the original 61 (a set drops duplicates): 557 books, 263.4 MB. Predicted ~350. New books averaged ~0.44 MB vs 0.7 MB for the originals, probably because the hand-picked ones were well known victorian era novels, all of which are HUGE. The random sample includes novellas, and story collections that no one has actually read.
- Kept the existing tokenizer instead of retraining because we want the same era, the same vocab, the same checkpoints, and retraining would change every token ID, making old checkpoints and val losses incomparable.
- Tokenized: 76.4M tokens in 32.5s (predicted 73.8M). 270,326 unique chunks (guessed 936K, then 155K). ~6x the text gave ~2.6x the unique chunks.
- Tokens per MB rose slightly (0.281M to 0.290M) because new and unfamiliar words were cut into more pieces.

 ## 2026-11-13
 ### Twelve vs Six Layer Tests
 - Jumping from 6 to 12 layers (using the same 557 books, 3 epochs, and full val set), Val loss dropped from 3.202 to 3.129
 - More data took the val loss from 3.735 to 3.202
 - This shows that more data provides better results than more depth.