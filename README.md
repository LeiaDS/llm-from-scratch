# llm-from-scratch#

A decoder-only transformer language model built from scratch in PyTorch, as part of an
Individual Learning Contract at The Evergreen State College (Fall 2026).

The goal is to understand every piece of a GPT-style model by building it by hand:
tokenization, masked self-attention, positional encoding, layer norm, and feedforward
blocks, without relying on pre-built model libraries. The model will be pretrained on a
Project Gutenberg corpus, fine-tuned with instruction tuning and LoRA, evaluated against
GPT-2 small, and deployed on constrained hardware.

## Status

Week 1: project setup, corpus selection, and byte-pair encoding tokenizer in progress.

## Roadmap

| Phase | Weeks | Focus |
| --- | --- | --- |
| Foundations | 1-3 | Tokenizer, data loader, embeddings, attention, full model |
| Pretraining | 4-5 | Training loop, pretraining run, loss tracking |
| Fine-tuning and evaluation | 6-8 | Instruction tuning, LoRA, comparison with GPT-2 small |
| Deployment and report | 9-10 | Quantization, hardware deployment, technical report |

## Repository layout

    tokenizer/        BPE tokenizer implementation
    model/            Transformer architecture
    data/             Corpus files (not tracked in git, see below)
    notes/            Working notes and design decisions
    checkpoints/      Model weights (not tracked in git)
    training_log.md   Running log of experiments, decisions, and results

## Data

The pretraining corpus is drawn from Project Gutenberg (public domain). Raw text and model
checkpoints are excluded from version control due to size. Instructions for downloading
and preprocessing the corpus will be added here once the pipeline is built.

## Acknowledgments

Technical foundation follows Sebastian Raschka's *Build a Large Language Model (From
Scratch)*. All implementation in this repository is my own work.

## Author

Leia Stretz