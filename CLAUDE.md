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

The kernel name in notebooks must be set to `python3` in the notebook JSON metadata — `.finetuning` is NOT a registered kernel on this machine.

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

### Qwen3-Specific Notes

- **Chat template**: Qwen3 uses ChatML format — `<|im_start|>role\n` ... `<|im_end|>`
- **`enable_thinking` parameter**: Pass to `apply_chat_template()` instead of `reasoning_effort`
  - `enable_thinking=False` for training and standard chat inference
  - `enable_thinking=True` for reasoning/thinking mode at inference
- **Empty `<think>` tags**: Even with `enable_thinking=False`, Qwen3's template adds `<think>\n\n</think>` before the response in training data — this is expected behavior
- **Response boundary tokens** for `train_on_responses_only()`:
  - `instruction_part = "<|im_start|>user\n"`
  - `response_part = "<|im_start|>assistant\n"`
- **Inference settings** (Qwen3 team recommendations):
  - Chat mode: `temperature=0.7, top_p=0.8, top_k=20`
  - Thinking mode: `temperature=0.6, top_p=0.95, top_k=20`
- **Kernel name**: Notebook metadata must use `"name": "python3"` (not `".finetuning"`) — only `python3`, `daytrader`, and `unsloth` kernels are registered on this machine
- **Batch size**: Qwen3-14B (4-bit, ~10.4 GB) leaves more headroom than GPT-OSS 20B, enabling `per_device_train_batch_size=2`
- **Training results** (1 epoch, Jordan Peterson books): ~321 steps, ~23 min, loss 2.44, peak VRAM 13.42 GB

### Comparison Notebook Design Patterns

- Both models are evaluated sequentially (can't fit two 20B models on 24GB simultaneously)
- Results are pickled to `comparison_cache/` between phases to avoid re-running inference
- Cache check at start of each model phase: if pkl exists, skip inference entirely
- `compute_text_stats()` must always append one value per text — never skip empty responses — or per-prompt plots will have length mismatches
- `compute_tfidf_similarity()` handles empty responses (returns 0.0) for fairness
- Fine-tuned model trained for 1 epoch (loss=3.01) produces mostly empty greedy-decode responses; this is expected and reflected honestly in the charts
- Key packages for analysis: `matplotlib`, `seaborn`, `wordcloud`, `nltk`, `scikit-learn`, `scipy`
