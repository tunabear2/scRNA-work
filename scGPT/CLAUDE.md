# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Install dependencies for H200 GPU (CUDA 12.1):
```bash
bash setup_scgpt.sh
```

The scgpt package is used as a local import (not installed via pip). Scripts at root level import from `./scgpt/` directly.

## Running Scripts

Each root-level script is a self-contained fine-tuning/inference pipeline. Run them from the project root:

```bash
python annotation.py        # Cell-type annotation (MS dataset)
python integration.py       # Batch integration (PBMC datasets)
python perturbation.py      # Perturbation prediction (Adamson/Norman)
python GRN_inference.py     # Gene Regulatory Network inference
python attentionGRN.py      # Attention-based GRN analysis
python multiomics_finetune.py  # Multi-omics (RNA + Protein)
python reference_mapping.py    # Reference cell mapping
```

## Running Tests

```bash
pytest tests/
pytest tests/test_tokenizer.py   # single test file
pytest tests/test_scformer.py    # model tests
```

## Data & Results Layout

Scripts expect pretrained model checkpoints under `data/models/` and write outputs to `results/<task>/`. Each task has a matching subdirectory in both `data/` and `results/`.

Download pretrained checkpoints from the Google Drive links in README.md and place them in the corresponding `data/models/` subdirectory.

## Architecture Overview

### Data Pipeline
```
AnnData → Preprocessor → GeneVocab (tokenizer) → DataCollator → TransformerModel
```

- **`scgpt/preprocess.py`** — `Preprocessor` class: normalizes counts, selects HVGs, bins expression values
- **`scgpt/tokenizer/gene_tokenizer.py`** — `GeneVocab`: maps gene names to integer token IDs; uses `default_gene_vocab.json` (16,906 genes)
- **`scgpt/data_collator.py`** — `DataCollator`: pads sequences, applies MLM-style masking
- **`scgpt/data_sampler.py`** — `SubsetsBatchSampler`: samples evenly across batches/studies

### Model
- **`scgpt/model/model.py`** — `TransformerModel`: main encoder; takes `(gene_ids, values)` pairs, outputs cell embeddings and gene expression predictions. Supports flash-attention, Domain-Specific Batch Normalization (DSBN), and multiple cell embedding modes (`cls`, `avg-pool`, `w-pool`).
- **`scgpt/model/multiomic_model.py`** — `MultiOmicTransformerModel`: extends base model for RNA + Protein joint modeling
- **`scgpt/model/generation_model.py`** — `TransformerGenerator`: decoder variant for generative tasks
- **`scgpt/model/grad_reverse.py`** — Gradient reversal layer used in adversarial discriminators (batch/domain adaptation)

### Training & Tasks
- **`scgpt/trainer.py`** — Training loop, evaluation logic, wandb integration
- **`scgpt/tasks/cell_emb.py`** — `get_batch_cell_embeddings()`, `embed_data()`: extract cell embeddings from a trained model
- **`scgpt/tasks/grn.py`** — `GeneEmbedding`: clusters gene embeddings via Louvain, builds gene programs, runs Enrichr pathway enrichment
- **`scgpt/utils/util.py`** — `load_pretrained()`, `set_seed()`, `eval_scib_metrics()`, plotting helpers

### scbank (Large-Scale Data)
- **`scgpt/scbank/`** — `DataBank`: disk-synced framework for managing multi-study datasets with lazy loading; used when datasets are too large for memory

## Key Patterns

- **Loading pretrained weights**: `load_pretrained(model, torch.load("path/to/ckpt.pt"))` from `scgpt.utils`
- **wandb**: scripts include a local shim so they run without a W&B account; set `USE_WANDB = False` at the top of each script to disable
- **Flash-attention**: optional; falls back to standard attention automatically via `load_pretrained`
- **DSBN**: Domain-Specific Batch Normalization is used across scripts for batch correction; requires `n_domain` parameter matching number of batches in the dataset
