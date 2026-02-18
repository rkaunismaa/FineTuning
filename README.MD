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
