# CSE 256 PA2: Speech Classification & Language Modeling

Transformer-based speech classification (3-way) and causal language modeling on presidential speech data.

**All experiment logs available on [Weights & Biases](https://wandb.ai/chentianyi453/CSE256_PA2_CLS) (CLS) and [CSE256_PA2_LM](https://wandb.ai/chentianyi453/CSE256_PA2_LM) (LM).**

## Quick Start

### Environment Setup

```bash
uv sync
source .venv/bin/activate

python
>>> import nltk
>>> nltk.download('punkt_tab')
```

### Train Models

```bash
uv run main.py
```



## Parameter Usage

### Training (all modes)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | both | Task: `cls`, `lm`, or `both` |
| `--model` | str | transformer | Model type (transformer) |
| `--batch_size` | int | 16 | Batch size |
| `--learning_rate` | float | 1e-3 | Learning rate |
| `--wandb` | flag | - | Enable Weights & Biases logging |
| `--wandb_project` | str | CSE256_PA2_CLS | WandB project name |
| `--wandb_run_name` | str | None | Custom run name for WandB |

### Transformer (shared encoder / decoder)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--block_size` | int | 32 | Maximum sequence length (context length) |
| `--n_embd` | int | 64 | Embedding dimension (d_model) |
| `--n_head` | int | 2 | Number of attention heads |
| `--n_layer` | int | 4 | Number of transformer layers |
| `--pe_type` | str | absolute | Positional encoding: `absolute` or `rope` |
| `--attn_type` | str | standard | Attention: `standard` or `deberta` (disentangled; CLS only) |
| `--theta` | float | 10000.0 | RoPE base (e.g. 10000) |

### Classification (CLS)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs_CLS` | int | 15 | Number of training epochs |
| `--cls_mode` | str | mean | Pooling: `mean` (mean over tokens), `cls_token` ([CLS] token), or `cls` |
| `--n_input` | int | 64 | Classifier input size (should match `n_embd`) |
| `--n_hidden` | int | 100 | Classifier hidden layer size |
| `--n_output` | int | 3 | Number of classes (3) |
| `--save_cls` | str | None | Path to save classifier state_dict |

### Language Modeling (LM)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_iters` | int | 500 | Max training iterations |
| `--eval_interval` | int | 100 | Evaluate perplexity every N iterations |
| `--eval_iters` | int | 100 | Number of batches for perplexity evaluation |
| `--save_lm` | str | None | Path to save LM state_dict |

## Model Overview

- **CLS:** Transformer encoder (token embedding + optional absolute/RoPE positional encoding, 4 layers of multi-head self-attention + FFN) → pooling (mean or [CLS] token) → 2-layer MLP classifier (64 → 100 → 3) with log-softmax; NLL loss.
- **LM:** Causal transformer decoder (same hyperparameters as encoder, autoregressive) trained with cross-entropy; metrics: train/val perplexity (including per-speaker val sets: H. Bush, Obama, W. Bush).

## Optimal Results (reference)

### Classification (Dev Accuracy)

| Configuration | Val Accuracy |
|---------------|------------------------|
| Baseline (absolute PE, mean pooling) | 85.3 |
| RoPE | 86.5 |
| Disentangled attention (DeBERTa-style) | 86.3 |
| [CLS] token pooling | 84.0 |

### Language Modeling


| Configuration | Val PPL | Val PPL (H. Bush) | Val PPL (Obama) | Val PPL (W. Bush) |
|---------------|---------|---------|---------|---------|
| Baseline | 429.8 | 410.3 | 345.9 | 429.8 |
