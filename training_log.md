# Training Log

## 2026-10-01
- Set up repo and project structure.
- Chose Project Gutenberg as the pretraining corpus because it's all public domain, and there's a large number of files to choose from. Issues may present in use of archaic language (though that should make hallucinations amusing), inconsistent formatting, and removal of headers and footers. Target corpus size is TBD.
- Downloaded Frankenstein as test text.
-BPE core (text_to_bytes, get_pair_counts, merge) implemented
- Initial merge used a for loop, which reused tokens across overlapping pairs ("aaa" became 256, 256). Switched to a while loop to skip both tokens after a match. Caught it because output length didn't shrink.