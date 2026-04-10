### Exploring Fine Tuning with Unsloth

This repository contains Jupyter notebooks for fine-tuning large language models using [Unsloth](https://github.com/unslothai/unsloth), an open-source library that provides 2x faster training with significantly less VRAM than standard approaches.

## Claude Prompt

The initial prompt submitted to Claude Opus 4.6:


    Create a notebook the explores finetuning a large language model using Unsloth. The Unsloth repository can be found at https://github.com/unslothai/unsloth. The creators of Unsloth have a repository of notebooks that demonstrate how to properly use Unsloth for many different models and use cases. 
    The model I want you to focus on can be found in the notebook https://github.com/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb. I want you to use this notebook as your guide on how to properly fine tune the "unsloth/gpt-oss-20b-unsloth-bnb-4bit" model. 

    The fine tuning data consists of books written by Jordan Peterson all in pdf format. These books can be found in the "FineTuning/Books/JordanPeterson" sub folder. I want you to create a notebook in the "FineTuning/NoteBooks/JordanPeterson" sub folder that reads from all of the books to fine tune the "unsloth/gpt-oss-20b-unsloth-bnb-4bit" model to create a new fine tuned model. 

    I know very little about how to do this, so be very detailed in explaining everything you do and why you did it. 

    The python environment '.finetuning' has been created using uv for all of this work and already has unsloth installed to it, plus other resources that may be needed. If any other resources are needed for this effort, use this environment to install those packages. 
    The 'FineTuning' folder is a github repository that has been cloned from 'git@github.com:rkaunismaa/FineTuning.git'


## Setup

The project uses a `uv`-managed Python 3.12 virtual environment (`.finetuning`) with Unsloth and its dependencies pre-installed.

**Hardware**: NVIDIA RTX 4090 (24GB VRAM)

## Notebooks

| Notebook | Model | Data Source | Description |
|----------|-------|-------------|-------------|
| [GPT-OSS 20B Jordan Peterson — Fine-Tuning](NoteBooks/JordanPeterson/GPT_OSS_20B_JordanPeterson_FineTuning.ipynb) | `unsloth/gpt-oss-20b-unsloth-bnb-4bit` | Jordan Peterson's books (4 PDFs) | Fine-tunes GPT-OSS 20B using LoRA on text extracted from Peterson's writings. Covers PDF extraction, dataset creation, training, inference, and model saving. |
| [GPT-OSS 20B Jordan Peterson — Comparison](NoteBooks/JordanPeterson/GPT_OSS_20B_JordanPeterson_Comparison.ipynb) | Base vs. fine-tuned | Same 4 PDFs (held-out passages) | Quantitative comparison between the base and fine-tuned models across 6 metrics (perplexity, TF-IDF similarity, keyword density, TTR, response length, word clouds) with 7 visualizations. |
| [Qwen3-14B Jordan Peterson — Fine-Tuning](NoteBooks/JordanPeterson/Qwen3_14B_JordanPeterson_FineTuning.ipynb) | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | Jordan Peterson's books (4 PDFs) | Fine-tunes Qwen3-14B using LoRA. Demonstrates Qwen3's ChatML format, `enable_thinking` parameter, and dual inference modes (chat + thinking). ~23 min training, loss 2.44. |
| [Qwen3-14B Jordan Peterson — Comparison](NoteBooks/JordanPeterson/Qwen3_14B_JordanPeterson_Comparison.ipynb) | Base vs. fine-tuned | Same 4 PDFs (held-out passages) | Quantitative comparison of Qwen3-14B base vs. fine-tuned across 6 metrics. Perplexity drops 33% (strong domain adaptation); other metrics reflect 1-epoch training artifacts. 7 visualizations. |
| [All Models — Cross-Architecture Comparison](NoteBooks/JordanPeterson/AllModels_JordanPeterson_Comparison.ipynb) | GPT-OSS 20B (base + FT) vs. Qwen3-14B (base + V1 + V4) | Same 4 PDFs (held-out passages) | Full cross-architecture comparison of all 5 model variants side-by-side. Covers intra-architecture fine-tuning effectiveness, inter-architecture base and fine-tuned comparisons, and training efficiency analysis. 7 visualizations. |
| [All Models — Cross-Architecture Comparison V2](NoteBooks/JordanPeterson/AllModels_JordanPeterson_Comparison_V2.ipynb) | GPT-OSS 20B (base + FT) vs. Qwen3-14B (base + V1 + V4) vs. Gemma 4 E4B | Same 4 PDFs (held-out passages) | Extends the 5-model comparison to include Gemma 4 E4B as a 6th variant. Adds Phase 6 inference cells, `generate_response_gemma4` helper, and a Gemma 4 column in all metric summaries and figures. 7 visualizations (2×3 word cloud grid now fully populated). |
| [Gemma 4 E4B Jordan Peterson — Fine-Tuning](NoteBooks/JordanPeterson/Gemma4_E4B_JordanPeterson_FineTuning.ipynb) | `unsloth/gemma-4-E4B-it` | V4 Q&A cache (3,936 pairs, front+back-matter removed) | Fine-tunes Gemma 4 E4B (Google's MoE model, ~4B effective active params) using the new `FastModel` API. Uses the same V4 training pipeline as Qwen3-14B V4 (3 epochs, r=16 LoRA). Includes an upgrade cell for Unsloth/Transformers ≥ 5.5.0 required by Gemma 4. Expected training loss: 13–15 (normal for small MoE models). |
| [Qwen3-14B Jordan Peterson — Fine-Tuning V2](NoteBooks/JordanPeterson/Qwen3_14B_JordanPeterson_V2_FineTuning.ipynb) | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | Synthetic Q&A pairs from 4 PDFs (generated by Claude Haiku) | Version 2 fine-tuning: addresses V1's passage-regurgitation problem with synthetic Q&A training data, 3 epochs, and LoRA r=32. Includes detailed post-mortem of V1 failures and full technical rationale for every hyperparameter. |
| [Qwen3-32B Jordan Peterson — Fine-Tuning](NoteBooks/JordanPeterson/Qwen3_32B_JordanPeterson_FineTuning.ipynb) | `unsloth/Qwen3-32B-unsloth-bnb-4bit` | Synthetic Q&A pairs from 4 PDFs (shared cache with V2) | Scales the V2 approach to Qwen3's 32B dense model. Uses conservative VRAM settings (max_seq_length=1024, batch_size=1, grad_accum=8) to fit within 24 GB. Includes pre-load VRAM check, OOM handler with fallback options, and a conclusions section with instructions for adding 32B to the AllModels comparison. |
| [Qwen3-14B Jordan Peterson — Fine-Tuning V3](NoteBooks/JordanPeterson/Qwen3_14B_JordanPeterson_V3_FineTuning.ipynb) | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | Cleaner Q&A cache from DataPrep notebook (4,867 pairs, front-matter removed) | Version 3: standalone fine-tuning notebook (no dataset generation). Trains on the DataPrep notebook's cleaner Q&A cache. Same hyperparameters as V2 (r=32, 3 epochs). Results: 1,827 steps, loss 1.5258, 136.9 min, 15.3 GB peak VRAM. |
| [Qwen3-14B Jordan Peterson — Fine-Tuning V4](NoteBooks/JordanPeterson/Qwen3_14B_JordanPeterson_V4_FineTuning.ipynb) | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | Cleanest Q&A cache from DataPrep notebook (3,936 pairs, front+back-matter removed) | Version 4: trains on the DataPrep notebook's cleanest cache with both front-matter AND back-matter (indexes, endnotes, bibliography) removed. Results: 1,476 steps, loss 1.6421, 98.0 min, 13.9 GB peak VRAM. 0/5 eval prompts trigger index output (V3 had 3/5) — back-matter removal directly confirmed effective. |
| [Qwen3.5-27B Jordan Peterson — Fine-Tuning](NoteBooks/JordanPeterson/Qwen3_5_27B_JordanPeterson_FineTuning.ipynb) | `unsloth/Qwen3.5-27B` | V4 Q&A cache | **OOM on RTX 4090** — 27B at 4-bit (~17 GB) exceeds available VRAM for training. Kept for reference; use the 9B notebook instead. |
| [Qwen3.5-9B Jordan Peterson — Fine-Tuning](NoteBooks/JordanPeterson/Qwen3_5_9B_JordanPeterson_FineTuning.ipynb) | `unsloth/Qwen3.5-9B` | V4 Q&A cache (3,936 pairs, front+back-matter removed) | Fine-tunes Qwen3.5-9B, a hybrid architecture (Gated DeltaNet + full attention, 32 layers). Loads via `FastVisionModel` with vision layers frozen. 8.5 GB 4-bit with ample VRAM headroom (batch_size=2, grad_accum=4). Requires unsloth>=2026.3.3 and transformers>=5.3.0. |
| [Qwen3-14B All Versions — Comparison](NoteBooks/JordanPeterson/Qwen3_14B_AllVersions_JordanPeterson_Comparison.ipynb) | Base vs V1 vs V2 vs V3 vs V4 | Same 4 PDFs (held-out passages) | Compares all five Qwen3-14B checkpoints side by side. Answers whether Q&A training (V1→V2), front-matter cleanup (V2→V3), and back-matter removal (V3→V4) produce measurable improvements. 8 visualizations including a version-progression line chart. Key finding: back-matter removal eliminated index contamination (3/10 → 0/10 prompts); cleaner data improves TF-IDF and TTR despite 931 fewer training pairs. |

## Project Structure

```
FineTuning/
├── Books/                          # Training data (PDFs, gitignored)
│   └── JordanPeterson/             # 4 books by Jordan Peterson
├── NoteBooks/                      # Fine-tuning notebooks
│   └── JordanPeterson/             # Peterson-focused fine-tuning
│       ├── outputs/                # Saved LoRA adapters (gitignored)
│       ├── qa_dataset/             # Synthetic Q&A cache (gitignored)
│       ├── comparison_cache*/      # Inference pkl caches (gitignored)
│       └── comparison_figures*/    # Generated chart PNGs
├── .gitignore
├── CLAUDE.md                       # AI assistant context file
└── README.MD                       # This file
```

## Approach

### Fine-Tuning Workflow

#### V1 — Passage Completion (GPT-OSS 20B, Qwen3-14B)

1. **Extract text** from source PDFs using PyMuPDF
2. **Chunk** into ~350-word passages with 50-word overlap
3. **Format** as passage-completion conversations: `(user: passage fragment) → (assistant: continuation)`
4. **Load** a 4-bit quantized model via Unsloth's `FastLanguageModel`
5. **Add LoRA adapters** (r=16) for parameter-efficient fine-tuning
6. **Train** 1 epoch using `SFTTrainer` with response-only masking
7. **Test** and **save** the LoRA adapter

**V1 limitation**: training on passage completion teaches the model to *continue* book text, not to *answer questions*. After 1 epoch the model regurgitates passages instead of generating coherent responses.

#### V2 — Synthetic Q&A (Qwen3-14B V2)

1. **Extract text** from PDFs — same as V1
2. **Generate questions** using Claude Haiku API: 2 questions per passage that the passage answers (~$1–3 total)
3. **Format** as instruction pairs: `(system: Peterson persona) + (user: question) + (assistant: verbatim passage)`
4. **Load** Qwen3-14B with LoRA r=32 (doubled capacity vs V1)
5. **Train** for **3 epochs** — crosses from memorisation into stylistic generalisation
6. **Test** and **save** the adapter to `outputs/qwen3_14b_peterson_v2_lora/`

The key insight: the *questions* are synthetic (generated by Claude Haiku) but the *answers* are verbatim Peterson — authentic, unmodified text from his books. The training task now matches the inference task.

#### V3 — Cleaner Data (Qwen3-14B V3)

V3 separates dataset preparation from fine-tuning and trains on the cleaner output of the
standalone `JordanPeterson_DataPrep.ipynb` notebook (front-matter pages removed):

1. **Run DataPrep first** to generate `qa_dataset/peterson_qa.jsonl` (4,867 pairs)
2. **Load** the cache — no PDF extraction or question generation in this notebook
3. **Train** with identical hyperparameters to V2 (r=32, 3 epochs, 2e-4 LR)
4. **Save** adapter to `outputs/qwen3_14b_peterson_v3_lora/`

**What changed vs V2**: The dataset shrank by 162 pairs (5,029 → 4,867), all from boilerplate
pages. The biggest reduction was We Who Wrestle with God (−103 pairs) and 12 Rules for Life
(−48 pairs); Maps of Meaning was nearly unchanged (−5 pairs).

**Results**: Loss 1.5258 (vs V2's 1.5058 — negligible difference), 136.9 min (vs 144.0 min),
peak VRAM 15.3 GB (vs 15.5 GB). Training metrics are essentially identical to V2.

**Inference quality**: Four of five evaluation prompts produced substantive, on-topic passages
from Peterson's works — appropriate content but still passage-like rather than direct answers.
One prompt (personal responsibility) triggered an **index page** response, revealing that
back-matter content (book indexes) is still present in the training data. Front-matter removal
alone is insufficient; a back-matter removal step (detecting and dropping index/bibliography
pages) is the next data-quality improvement needed.

**Key lesson**: The largest remaining source of non-content training data is not front matter
but back matter — index entries, figure captions, and bibliography pages that exist throughout
and at the end of the books.

#### V4 — Front + Back-Matter Removed (Qwen3-14B V4)

V4 extends V3 by also removing back-matter pages (indexes, endnotes, bibliography) from the
dataset using a new heuristic in `JordanPeterson_DataPrep.ipynb`:

1. **Run updated DataPrep** to generate `qa_dataset/peterson_qa.jsonl` (3,936 pairs — both
   front-matter AND back-matter pages removed)
2. **Load** the cache — no PDF extraction or question generation in this notebook
3. **Train** with identical hyperparameters to V3 (r=32, 3 epochs, 2e-4 LR)
4. **Save** adapter to `outputs/qwen3_14b_peterson_v4_lora/`

**What changed vs V3**: Dataset shrank by 931 pairs (4,867 → 3,936). Back-matter detection
uses two signals: (1) ≥5 page-range patterns AND numeric density > 5%; (2) numeric density > 40%.

**Results**: Loss 1.6421 (higher than V3's 1.5258 — expected with smaller dataset), 98.0 min
(vs 136.9 min), peak VRAM 13.9 GB (vs 15.3 GB).

**Inference quality**: 0/5 evaluation prompts trigger index output (V3 had 3/5). All responses
are substantive Peterson prose. Back-matter removal directly confirmed effective. Still in
verbatim-retrieval mode — none of the fine-tuned versions truly answers questions.

#### Gemma 4 E4B — Google MoE Architecture

Google's Gemma 4 was released April 2026.  The E4B variant uses a Mixture-of-Experts
architecture with ~4B *effective* active parameters per forward pass, drawn from a
larger total parameter pool.

**Requires updated libraries before first run:**
```bash
uv pip install --upgrade unsloth unsloth-zoo "transformers>=5.5.0"
```
The fine-tuning notebook contains a setup cell that does this automatically.

1. **Run DataPrep** to generate `qa_dataset/peterson_qa.jsonl` (same V4 cache)
2. **Load** via `FastModel` (Gemma 4's unified API, handles text and multimodal)
3. **Apply LoRA** with `finetune_vision_layers=False`, `finetune_language_layers=True`
4. **Train** with `batch_size=1`, `grad_accum=8` (effective batch=8), r=16, 3 epochs
5. **Save** adapter to `outputs/gemma4_e4b_peterson_v1_lora/`

**Key differences from Qwen3 notebooks:**
- Uses `FastModel` API (not `FastLanguageModel` or `FastVisionModel`)
- Chat format: `<start_of_turn>user / <start_of_turn>model` (same as Gemma 3)
- System prompt is prepended to the first user turn (not a separate turn)
- `train_on_responses_only`: instruction=`"<start_of_turn>user\n"`, response=`"<start_of_turn>model\n"`
- No `enable_thinking` parameter

**Expected training loss**: 13–15 (normal for E2B/E4B MoE — not a failure signal).
The 26B/31B variants show 1–3; use those only for same-architecture comparison.

**VRAM**: ~10 GB weights (4-bit), ~17 GB peak training — comfortable on 24 GB RTX 4090.

**Library requirements**: `unsloth>=2026.4.x`, `transformers>=5.5.0`.

#### Qwen3.5-9B — New Architecture (Hybrid Linear + Full Attention)

Qwen3.5 (Feb 2026) introduces a fundamentally different architecture from Qwen3.
The 27B variant was attempted first but **OOM'd on the RTX 4090** (~17 GB weights
at 4-bit leaves insufficient headroom for training).  The 9B variant fits
comfortably:

1. **Run DataPrep** to generate `qa_dataset/peterson_qa.jsonl` (same V4 cache)
2. **Load** via `FastVisionModel` (all Qwen3.5 models are VLMs)
3. **Apply LoRA** with `finetune_vision_layers=False`, `finetune_language_layers=True`
4. **Train** with `batch_size=2`, `grad_accum=4` (effective batch=8), same r=32, 3 epochs
5. **Save** adapter to `outputs/qwen3_5_9b_peterson_lora/`

**Key architecture differences from Qwen3:**
- 32 layers: 24 linear attention (Gated DeltaNet, O(n) complexity) + 8 full attention
- Hidden dim 4096, 16 attention heads, 4 KV heads
- Vision encoder included (frozen for text-only training)
- Same ChatML template (`<|im_start|>` / `<|im_end|>`)

**VRAM**: ~8.5 GB weights (4-bit), ~11-12 GB peak training — ample headroom on 24 GB.

**Library requirements**: `unsloth>=2026.3.3`, `transformers>=5.3.0`.

### All-Versions Comparison Results (Base vs V1 vs V2 vs V3 vs V4)

A full side-by-side evaluation of all four Qwen3-14B checkpoints across 10 evaluation prompts
and 8 Peterson-domain metrics reveals the following:

#### Quantitative Metrics

| Metric | Base | V1 | V2 | V3 | V4 |
|--------|------|----|----|-----|-----|
| Avg Perplexity (↓ better) | 19.1 | 12.8 (−33%) | **10.0 (−47%)** | 11.0 (−42%) | 11.3 (−41%) |
| Avg TF-IDF Similarity (↑ better) | **0.061** | 0.054 | 0.044 | 0.027 | 0.037 |
| Avg Keyword Density (↑ better) | **4.19%** | 3.30% | 2.26% | 2.62% | 1.79% |
| Avg Response Length | 204 words | 250 words | 218 words | 181 words | 238 words |
| Avg TTR | **0.661** | 0.380 | 0.551 | 0.586 | 0.629 |

#### What the Responses Actually Look Like

**Base model**: Well-structured markdown answers with headers and bullet points. Accurate and
coherent but generic — a textbook summary style, not Peterson's voice. Highest TF-IDF because
keyword-dense structured responses happen to match reference passage vocabulary.

**V1 (passage completion)**: Every response begins mid-sentence with `"the world,"` — the model
is continuing a sentence fragment, not answering a question. Generates repetitive loops (e.g.,
`"and then you can be at one with everything. And then you can be at one with nothing."` repeated
~15 times). The training task is completely visible at inference.

**V2 (Q&A, 3 epochs)**: Produces verbatim Peterson passages — philosophically rich, authentic
content from the actual books. The training task now matches the inference task so responses are
relevant, but the model is in retrieval mode rather than generation mode. Q6 (hero archetype)
triggers a raw **index page** (`102, 135, 146, 178...`) — back-matter contamination.

**V3 (Q&A, clean data, 3 epochs)**: Also verbatim passages but with *more* index contamination
than V2. Three prompts (Q2, Q6, Q7) trigger index/back-matter output. Paradoxically, removing
front matter exposed more back-matter entries proportionally in the training corpus.

**V4 (Q&A, cleanest data, 3 epochs)**: Verbatim passages, **zero index contamination** (0/10
prompts). TF-IDF (0.037) and TTR (0.629) both recover vs V3 despite 931 fewer training pairs —
cleaner data outweighs reduced volume. Still in retrieval mode.

#### Key Findings

1. **Perplexity improves strongly with training** — V2 achieves −47% vs base, confirming
   genuine domain adaptation at the language-model level.

2. **TF-IDF and keyword density decrease with training** — fine-tuning pulls responses away
   from Peterson's surface vocabulary. The base model's structured answers happen to use more
   of Peterson's keyword vocabulary than the fine-tuned models' verbatim passage reproduction.

3. **No model actually answers the questions** — all fine-tuned variants reproduce training
   passages (verbatim or near-verbatim). The models learned to retrieve, not to generate.
   Fixing this requires either: (a) more training epochs to push past retrieval into
   stylistic generalisation, or (b) DPO/RLHF alignment after SFT.

4. **Back-matter removal (V3→V4) directly eliminated index contamination** — 3/10 → 0/10
   prompts trigger index output. V4 TF-IDF and TTR improve over V3 despite fewer pairs,
   confirming that data quality outweighs data volume.

5. **V2 vs V3 are nearly identical** — despite losing 162 training pairs, training loss
   changed by only 0.02. The front-matter fix had minimal impact on final model behaviour.

### Comparison Workflow

#### Claude Prompt

The comparison prompt submitted to Claude Opus 4.6:

    Create a notebook that compares the performance of the "unsloth/gpt-oss-20b-unsloth-bnb-4bit" base model and the model fine tuned on the 4 books from Jordan Peterson. Use whatever quantitative measurements are available that will be able to visually display how the fine tuned model is more Jordan Peterson like than the base model. Be detailed in your explanations and provide visuals. 

The comparison notebook measures domain adaptation across six independent axes:

| Metric | What It Measures |
|--------|-----------------|
| **Perplexity** | How surprised the model is by Peterson's actual prose (lower = better) |
| **TF-IDF Cosine Similarity** | Vocabulary overlap between responses and real Peterson passages |
| **Keyword Density** | Frequency of Peterson's characteristic terms (chaos, order, meaning, hero…) |
| **Type-Token Ratio** | Vocabulary richness — fraction of unique words per response |
| **Response Length** | Average words per response (Peterson is verbose) |
| **Word Cloud** | Visual of which words dominate each model's responses |

Both models are loaded and evaluated sequentially (memory constraint: two 20B models can't fit in 24GB simultaneously). Results are cached to `comparison_cache/` as pickle files so inference only runs once.

### Training Results Summary

| Model | Data | Epochs | Steps | Time | Loss | Peak VRAM | Adapter |
|-------|------|--------|-------|------|------|-----------|---------|
| GPT-OSS 20B V1 | Passage completion (~2,519 passages) | 1 | 641 | 73.3 min | 3.01 | 13.8 GB | ~260 MB |
| Qwen3-14B V1 | Passage completion (~2,519 passages) | 1 | 321 | 23.3 min | 2.44 | 13.4 GB | ~260 MB |
| Qwen3-14B V2 | Synthetic Q&A (5,029 pairs, w/ front matter) | 3 | 1,887 | 144.0 min | 1.5058 | 15.5 GB | 513.9 MB |
| Qwen3-14B V3 | Synthetic Q&A (4,867 pairs, front-matter removed) | 3 | 1,827 | 136.9 min | 1.5258 | 15.3 GB | 513.9 MB |
| Qwen3-14B V4 | Synthetic Q&A (3,936 pairs, front+back-matter removed) | 3 | 1,476 | 98.0 min | 1.6421 | 13.9 GB | 513.9 MB |
| Qwen3.5-27B | Synthetic Q&A (V4 cache) | 3 | — | — | — | **OOM** | — |
| Qwen3.5-9B | Synthetic Q&A (3,936 pairs, V4 cache) | 3 | TBD | TBD | TBD | ~11-12 GB est. | TBD |
| Qwen3-32B | Synthetic Q&A (shared cache) | 3 | TBD | TBD | TBD | ~20–22 GB est. | TBD |
| Gemma 4 E4B V1 | Synthetic Q&A (3,936 pairs, V4 cache) | 3 | TBD | TBD | 13–15 (expected) | ~17 GB est. | TBD |

## References

- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning library
- [Unsloth Notebooks](https://github.com/unslothai/notebooks) - Official example notebooks
- [GPT-OSS 20B Fine-tuning Reference](https://github.com/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb) - Reference notebook used as the basis for our implementation
- [Qwen3 Fine-tuning Reference](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb) - Reference notebook used for Qwen3 implementation
- [Gemma 4 Fine-tuning Guide](https://unsloth.ai/docs/models/gemma-4/train) - Unsloth guide for Gemma 4
- [Gemma 4 E4B Reference Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Text.ipynb) - Reference notebook used for Gemma 4 implementation
