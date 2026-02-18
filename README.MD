### Exploring Fine Tuning with Unsloth

This repository contains Jupyter notebooks for fine-tuning large language models using [Unsloth](https://github.com/unslothai/unsloth), an open-source library that provides 2x faster training with significantly less VRAM than standard approaches.

## Setup

The project uses a `uv`-managed Python 3.12 virtual environment (`.finetuning`) with Unsloth and its dependencies pre-installed.

**Hardware**: NVIDIA RTX 4090 (24GB VRAM)

## Notebooks

| Notebook | Model | Data Source | Description |
|----------|-------|-------------|-------------|
| [GPT-OSS 20B Jordan Peterson](NoteBooks/JordanPeterson/GPT_OSS_20B_JordanPeterson_FineTuning.ipynb) | `unsloth/gpt-oss-20b-unsloth-bnb-4bit` | Jordan Peterson's books (4 PDFs) | Fine-tunes GPT-OSS 20B using LoRA on text extracted from Peterson's writings. Covers PDF extraction, dataset creation, training, inference, and model saving. |

## Project Structure

```
FineTuning/
├── Books/                  # Training data (PDFs, gitignored)
│   └── JordanPeterson/     # 4 books by Jordan Peterson
├── NoteBooks/              # Fine-tuning notebooks
│   └── JordanPeterson/     # Peterson-focused fine-tuning
├── .gitignore
├── CLAUDE.md               # AI assistant context file
└── README.MD               # This file
```

## Approach

Each notebook follows this general workflow:

1. **Extract text** from source PDFs using PyMuPDF
2. **Create a training dataset** by chunking text into passages and formatting as conversations
3. **Load a quantized model** (4-bit via bitsandbytes) using Unsloth's `FastLanguageModel`
4. **Add LoRA adapters** for parameter-efficient fine-tuning (~0.04% of parameters trained)
5. **Train** using `SFTTrainer` with response-only masking
6. **Test** the fine-tuned model with relevant prompts
7. **Save** the LoRA adapters for later use

## References

- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning library
- [Unsloth Notebooks](https://github.com/unslothai/notebooks) - Official example notebooks
- [GPT-OSS 20B Fine-tuning Reference](https://github.com/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb) - Reference notebook used as the basis for our implementation
