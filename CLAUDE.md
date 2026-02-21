# CLAUDE.md

## Project Overview

This repository explores fine-tuning large language models using [Unsloth](https://github.com/unslothai/unsloth). The primary focus is on parameter-efficient fine-tuning (LoRA) with 4-bit quantized models on consumer GPUs.

## Repository Structure

```
FineTuning/
├── Books/                          # Source training data (PDFs, gitignored)
│   └── JordanPeterson/             # Jordan Peterson's books (4 PDFs)
├── NoteBooks/
│   └── JordanPeterson/
│       ├── peterson_config.json    # All tunable params (paths, chunk sizes, backend)
│       ├── JordanPeterson_DataPrep.ipynb       # Step 1: extract + Q&A generation
│       ├── Qwen3_14B_JordanPeterson_V2_FineTuning.ipynb
│       ├── Qwen3_14B_JordanPeterson_V3_FineTuning.ipynb
│       ├── Qwen3_32B_JordanPeterson_FineTuning.ipynb
│       ├── AllModels_JordanPeterson_Comparison.ipynb
│       └── qa_dataset/             # Q&A cache (gitignored)
│           └── peterson_qa.jsonl
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
- `anthropic` - Anthropic Python SDK (used in V2 for synthetic Q&A dataset generation)

### Installing packages

```bash
uv pip install <package> --python /home/rob/PythonEnvironments/FineTuning/.finetuning/bin/python
```

## Notebooks

Notebooks are in `NoteBooks/` organized by data source.

### Notebook Execution Order (Jordan Peterson pipeline)

1. **`JordanPeterson_DataPrep.ipynb`** — Run this first. Extracts PDF text (with front-matter removal), generates Q&A pairs, writes `qa_dataset/peterson_qa.jsonl`.
2. **`Qwen3_14B_JordanPeterson_V2_FineTuning.ipynb`** — Fine-tunes Qwen3-14B on the cached Q&A pairs (5,029 pairs, V2 data — includes front matter).
3. **`Qwen3_14B_JordanPeterson_V3_FineTuning.ipynb`** — Fine-tunes Qwen3-14B on the cleaner DataPrep cache (4,867 pairs, no front matter). Output: `outputs/qwen3_14b_peterson_v3_lora/`.
4. **`Qwen3_32B_JordanPeterson_FineTuning.ipynb`** — Fine-tunes Qwen3-32B on the same cache.

Each fine-tuning notebook is also self-contained and runs top-to-bottom.

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
- **Gitignored**: `Books/`, `**/outputs/`, `**/unsloth_compiled_cache/`, `**/*_executed.ipynb`, `**/comparison_cache*/`, `**/qa_dataset/`, `__pycache__/`
- **Remote**: `git@github.com:rkaunismaa/FineTuning.git` (branch: `main`)
- Do not commit training outputs, checkpoints, or model weights

## Key Technical Details

### Data Preparation Notebook (`JordanPeterson_DataPrep.ipynb`)

Centralises PDF extraction and Q&A generation into one reusable notebook:
- **Config**: reads all parameters from `peterson_config.json` (paths, chunk sizes, backend)
- **Front-matter removal**: 3-tier heuristic skips publisher pages before Chapter 1
  - Tier 1: page with chapter-1 marker + ≥150 words + no chapter-2 marker
  - Tier 2: first page with ≥200 words after the last copyright/publisher page
  - Tier 3: no-op — start from page 0
- **Backend**: `"local"` (default, free) or `"anthropic"` (paid API) — set in config
- **Local model**: `unsloth/Qwen3-4B-unsloth-bnb-4bit` by default (~2.5 GB VRAM, ~45 min)
- **Cache-aware**: skips generation if `qa_dataset/peterson_qa.jsonl` is ≥90% complete
- **Cleanup**: unloads the generation model and frees VRAM before exiting

Key regexes (in `peterson_config.json` indirectly; defined in notebook):
- `_CHAPTER1_RE`: matches "Chapter 1", "Rule 1", "Rule I", "Overture", "Cain and Abel"
- `_CHAPTER2_RE`: matches "Chapter 2", "Rule 2", "Rule II", "Noah" (TOC filter)
- `_FRONTMATTER_RE`: matches ISBN, copyright, "All rights reserved", publisher domains

### GPT-OSS Chat Template

The GPT-OSS model uses OpenAI's Harmony format. When formatting simple conversations (no multi-channel reasoning), the chat template produces:
- `<|start|>assistant<|message|>` (NOT `<|start|>assistant<|channel|>final<|message|>`)
- The `response_part` for `train_on_responses_only()` must match the actual template output
- The notebook auto-detects the correct token by inspecting the formatted data

### Training Data from PDFs (V1 approach)

- PDFs are extracted with PyMuPDF (`fitz`)
- Text is cleaned (non-printable chars, page numbers, excess whitespace)
- Chunked into ~350-word passages with 50-word overlap
- Formatted as system/user/assistant conversations where user = passage fragment, assistant = continuation
- Token counts typically 550-1100 per example with 2048 max_seq_length
- **Limitation**: trains a passage-completion task, causing regurgitation at inference (see V2)

### V2 Fine-Tuning: Synthetic Q&A Dataset

The V2 approach (see `Qwen3_14B_JordanPeterson_V2_FineTuning.ipynb`) replaces raw passage completion with synthetic Q&A pairs:

1. **Extract** the same ~350-word passage chunks from PDFs
2. **Generate questions** via Claude Haiku API: 2 questions per passage that the passage directly answers (~$1–3 total for all 4 books)
3. **Format** as `(system: Peterson persona) + (user: question) + (assistant: passage verbatim)`
4. **Train** for 3 epochs with r=32 LoRA

Key files:
- `qa_dataset/peterson_qa.jsonl` — cached Q&A pairs (gitignored, regenerated automatically)
- `outputs/qwen3_14b_peterson_v2_lora/` — V2 adapter weights (gitignored)

**Why this matters**: the training task now matches the inference task (answer a question) rather than teaching the model to continue passages. This is the root fix for the passage-regurgitation problem observed in V1.

### V3 Fine-Tuning: Cleaner DataPrep Dataset

`Qwen3_14B_JordanPeterson_V3_FineTuning.ipynb` — fine-tuning only (no dataset generation).

- Reads from `qa_dataset/peterson_qa.jsonl` produced by `JordanPeterson_DataPrep.ipynb`
- Dataset: **4,867 pairs** (vs V2's 5,029) — front-matter pages removed by DataPrep heuristic
- Same hyperparameters as V2: r=32, alpha=32, 3 epochs, 2e-4 LR
- Output: `outputs/qwen3_14b_peterson_v3_lora/` (does NOT overwrite V2)
- Adapter size: 513.9 MB (identical to V2 — same r=32 architecture)

**Actual V3 results (run 2026-02-21):**
- Steps: 1,827 | Loss: 1.5258 | Time: 136.9 min | Peak VRAM: 15.3 GB
- Loss is +0.02 vs V2 (1.5058) — negligible, within noise; fewer training pairs means slightly
  less gradient signal but the difference is immaterial

**Actual dataset reduction (different from predicted):**
| Book | V2 pairs | V3 pairs | Δ |
|------|----------|----------|---|
| Maps of Meaning | 1,826 | 1,821 | -5 |
| We Who Wrestle with God | 1,367 | 1,264 | **-103** |
| 12 Rules for Life | 974 | 926 | -48 |
| Beyond Order | 862 | 856 | -6 |

The biggest reduction was in We Who Wrestle (-103), not Maps of Meaning (-5) as predicted.
Maps of Meaning's front matter was shorter than expected by the DataPrep heuristic.

**Inference quality (5 eval prompts, greedy decoding):**
- Q1, Q3, Q4, Q5: Substantive philosophical passages in Peterson's voice — content-appropriate
  but still passage-like rather than directly answering the question
- **Q2 produced raw index text** ("143–74 and Genesis story, 160–68 in Harry Potter series,
  259–60…") — confirms that back-matter content (index pages) is still present in the training
  data; front-matter removal alone does not eliminate this contamination
- The sample training example in Part 4 showed a figure-list entry ("FIGURES 1 The Domain
  and Constituent Elements of the Known 15 2 The Metamythological Cycle of the Way 17…")
  — figure lists within the book body are not touched by front-matter removal

**V3 conclusion:** Nearly identical to V2 in both training metrics and inference quality.
The front-matter fix improved dataset purity but the remaining noise (figure lists, indexes)
suggests that a back-matter removal step is the next highest-leverage data improvement.

### V2 vs V1 Hyperparameter Comparison

| Parameter | V1 | V2 | V3 |
|-----------|----|----|-----|
| Training data | Passage completion | Synthetic Q&A (w/ front matter) | Synthetic Q&A (front-matter removed) |
| Q&A pairs | N/A | 5,029 | 4,867 |
| Steps | 321 | 1,887 | 1,827 |
| Loss | 2.44 | 1.5058 | 1.5258 |
| Time | 23.3 min | 144.0 min | 136.9 min |
| Peak VRAM | 13.4 GB | 15.5 GB | 15.3 GB |
| Epochs | 1 | 3 | 3 |
| LoRA rank | r=16 | r=32 | r=32 |
| Adapter size | ~260 MB | 513.9 MB | 513.9 MB |

### Q&A Generation with Claude API

- Model: `claude-haiku-4-5-20251001` (cost-efficient, sufficient for question generation)
- Prompt: asks for 2 questions per passage as a JSON array of strings (answers NOT in output — reuse passage text)
- Retry logic: 3 attempts with exponential backoff on API errors or malformed JSON
- Incremental save: appends to JSONL after each passage so interruptions are safe
- Cache check: skip generation if file covers ≥90% of passages
- Requires `ANTHROPIC_API_KEY` environment variable (or `~/.env` file)

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

### All-Models Comparison Notebook

`AllModels_JordanPeterson_Comparison.ipynb` compares all 4 variants (GPT-OSS base/tuned + Qwen3 base/tuned) side by side:
- Uses `comparison_cache_all_models/` with files named `{key}_results.pkl` (e.g. `qwen3_tuned_results.pkl`)
- Existing single-model pkl files are format-compatible and can be copied to bootstrap the cache
- Two separate inference wrappers: `generate_response_gptoss()` and `generate_response_qwen3()`
- To add a new model variant (e.g. V2 fine-tuned): update `MODEL_KEYS`, `MODEL_PATHS`, `MODEL_COLORS`, and delete the relevant pkl to force re-inference

### Qwen3-32B Fine-Tuning Notebook

`Qwen3_32B_JordanPeterson_FineTuning.ipynb` — scales the V2 synthetic Q&A approach to the 32B dense model:
- **VRAM budget**: 32B in 4-bit = ~17.6 GB weights alone; peak target < 24 GB
- **Conservative settings**: `max_seq_length=1024` (vs 2048), `batch_size=1` (vs 2), `grad_accum=8` (effective batch=8, same as V2)
- **LoRA**: r=32, alpha=32 — same as V2
- **Epochs**: 3 — same as V2
- **Shared Q&A cache**: reuses `./qa_dataset/peterson_qa.jsonl` from V2 (no regeneration needed)
- **OOM handler**: catches `RuntimeError` containing "out of memory" or "cuda"; prints three explicit fallback options:
  1. Reduce `max_seq_length` to 768 and retry
  2. Switch to `unsloth/Qwen3-30B-A3B-bnb-4bit` (MoE, lighter but less style-consistent)
  3. Fall back to Qwen3-14B V2
- **Pre-load VRAM check**: warns if free VRAM < 17 GB before model load
- **Output**: `./outputs/qwen3_32b_peterson_lora/`
- **Why dense over MoE**: Qwen3-30B-A3B routes tokens to different experts — inconsistent style learning. Dense 32B applies the same weights to every token, making consistent stylistic imitation easier.
- **Conclusions cell**: exact code snippet for adding 32B to `AllModels_JordanPeterson_Comparison.ipynb`
