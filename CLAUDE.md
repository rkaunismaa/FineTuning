# CLAUDE.md

## Project Overview

This repository explores fine-tuning large language models using [Unsloth](https://github.com/unslothai/unsloth). The primary focus is on parameter-efficient fine-tuning (LoRA) with 4-bit quantized models on consumer GPUs.

## Repository Structure

```
FineTuning/
├── Books/                  # Source training data (PDFs, gitignored)
│   └── JordanPeterson/     # Jordan Peterson's books (4 PDFs)
├── NoteBooks/              # Jupyter notebooks for fine-tuning
│   └── JordanPeterson/     # Fine-tuning on Peterson's writings
├── .gitignore
├── CLAUDE.md               # This file
└── README.MD
```

## Environment

- **Python environment**: `.finetuning` virtual environment located at `/home/rob/PythonEnvironments/FineTuning/.finetuning/`
- **Created with**: `uv`
- **Python version**: 3.12
- **GPU**: NVIDIA RTX 4090 (24GB VRAM), secondary RTX 2070 SUPER (8GB)

### Key packages

- `unsloth` - Fast fine-tuning library (2x speedup, VRAM reduction)
- `torch` - PyTorch with CUDA 12.8
- `transformers` - HuggingFace model loading and tokenization
- `trl` - SFTTrainer for supervised fine-tuning
- `peft` - LoRA adapter management
- `datasets` - HuggingFace dataset handling
- `pymupdf` (fitz) - PDF text extraction

### Installing packages

```bash
uv pip install <package> --python /home/rob/PythonEnvironments/FineTuning/.finetuning/bin/python
```

## Notebooks

Notebooks are in `NoteBooks/` organized by data source. Each notebook is self-contained and runs top-to-bottom.

### Running notebooks

```bash
# Execute via nbconvert (headless)
cd NoteBooks/JordanPeterson
/home/rob/PythonEnvironments/FineTuning/.finetuning/bin/jupyter nbconvert \
  --to notebook --execute --ExecutePreprocessor.timeout=7200 \
  --output <name>_executed.ipynb <name>.ipynb
```

The kernel name in notebooks is set to `.finetuning`.

## Git Conventions

- **Commit style**: Short summary line, blank line, details
- **Gitignored**: `Books/`, `**/outputs/`, `**/unsloth_compiled_cache/`, `**/*_executed.ipynb`, `__pycache__/`
- **Remote**: `git@github.com:rkaunismaa/FineTuning.git` (branch: `main`)
- Do not commit training outputs, checkpoints, or model weights

## Key Technical Details

### GPT-OSS Chat Template

The GPT-OSS model uses OpenAI's Harmony format. When formatting simple conversations (no multi-channel reasoning), the chat template produces:
- `<|start|>assistant<|message|>` (NOT `<|start|>assistant<|channel|>final<|message|>`)
- The `response_part` for `train_on_responses_only()` must match the actual template output
- The notebook auto-detects the correct token by inspecting the formatted data

### Training Data from PDFs

- PDFs are extracted with PyMuPDF (`fitz`)
- Text is cleaned (non-printable chars, page numbers, excess whitespace)
- Chunked into ~350-word passages with 50-word overlap
- Formatted as system/user/assistant conversations
- Token counts typically 550-1100 per example with 2048 max_seq_length
