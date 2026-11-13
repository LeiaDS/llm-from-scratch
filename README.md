# llm-from-scratch

A decoder-only transformer language model built from scratch in PyTorch, as part of an
Individual Learning Contract at The Evergreen State College (Fall 2026).

The goal is to understand every piece of a GPT-style model by building it by hand:
tokenization, masked self-attention, positional embeddings, layer norm, and feedforward
blocks, without relying on pre-built model libraries. The model is being pretrained on
19th-century novels from Project Gutenberg, then fine-tuned with instruction tuning,
evaluated against GPT-2 small, and deployed on constrained hardware.

## Status

- Weeks 1 to 3 complete: BPE tokenizer, data loader, and full model (6 layers, 25.5M
  parameters), verified with shape tests and an overfit test.
- Corpus tokenized, 12.1M tokens
- Weeks 4 and 5 complete. 
- 557 books used for corpus, 12 layers (about 44M parameters) being used instead of 6. 76.4 mil tokens. Best full fal loss was 3.129.
- NEXT - fine tuning the model, begin comparing it to GPT-2 small.

All working notes, results, and the reasoning behind each decision are in
training_log.md.

## Roadmap

| Phase | Weeks | Focus |
| --- | --- | --- |
| Foundations | 1-3 | Tokenizer, data loader, embeddings, attention, full model |
| Pretraining | 4-5 | Corpus, training loop, pretraining run, loss tracking |
| Fine-tuning and evaluation | 6-8 | Instruction tuning, comparison with GPT-2 small (LoRA as a stretch goal) |
| Deployment and report | 9-10 | Quantization, hardware deployment, technical report |

## Repository layout

    tokenizer/        BPE tokenizer, trained merges (merges.json), round-trip test
    model/            Transformer architecture (attention, layer norm, feedforward, blocks, full GPT)
    training/         Data loader and overfit test
    scripts/          Corpus download, tokenizer training, vocabulary inspection
    data/             Corpus files (not tracked in git)
    checkpoints/      Model weights (not tracked in git)
    training_log.md   Running log of decisions, results, and working notes

## Setup

Requires Python 3.12 and PyTorch with CUDA (developed on an RTX 5070 Ti). Run everything
from the project root.

    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install numpy truststore regex

    python scripts/download_gutenberg.py      # download and clean the 61-book corpus
    python -m scripts.train_tokenizer         # train the tokenizer (about 15 minutes)

## Tests

    python tokenizer/bpe.py                   # toy tokenizer round trip
    python -m model.gpt                       # full model shape test and parameter count
    python -m training.dataset                # data loader test
    python -m training.overfit                # overfit a single batch (needs a CUDA GPU)

## Data

The corpus is 61 novels originally written in English, roughly 1810 to 1900, from
Project Gutenberg (public domain). The download script strips Gutenberg's license
header and footer from each book. Raw text and model checkpoints are excluded from
version control due to size.

## Acknowledgments

Technical foundation follows Sebastian Raschka's *Build a Large Language Model (From
Scratch)*. encode() in the tokenizer is adapted from Andrej Karpathy's minbpe.

I used Claude (Anthropic) as a tutor throughout: it explained concepts, provided code
skeletons and some complete scaffolding (such as the overfit test and the data download
and tokenizer training scripts), and reviewed my work. I wrote or completed the core
implementations myself and documented my understanding of each component in
training_log.md.

## Author

Leia Stretz