# OPSA Training Example

## Overview

This directory contains scripts for training models with the **OPSA (On-Policy Self-Distillation for Safety Alignment)** method.

OPSA reduces the "safety tax" in LLM alignment by using on-policy self-distillation with type-conditional privileged contexts, concentrating gradient updates on safety-critical tokens in the early "refusal-decision window."

**Paper:** [Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation](https://arxiv.org/abs/2605.15239)
**Code:** [https://github.com/FYYFU/OPSA](https://github.com/FYYFU/OPSA)

---

## Quick Start

### 1. Install EasyOPD

```bash
pip install -e .
```

### 2. Prepare Safety Data

```bash
# Download and process SafeChain dataset
python examples/opsa/prepare_safety_data.py \
    --dataset UWNSL/SafeChain \
    --output_dir data/opsa/

# Or use per-model ThinkSafe dataset
python examples/opsa/prepare_safety_data.py \
    --dataset Seanie-lee/ThinkSafe-Qwen3-1.7B \
    --output_dir data/opsa/
```

### 3. Run Training

```bash
# Default configuration (Qwen3-1.7B, SafeChain)
bash examples/opsa/run_opsa.sh

# Custom model and data paths
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
DATA_PATH=data/opsa/thinksafe_qwen3_1.7b_train.parquet \
bash examples/opsa/run_opsa.sh
```

---

## Configuration

### Key Parameters

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| Model | `MODEL_PATH` | `Qwen/Qwen3-1.7B` | Base model (also used as teacher) |
| Data | `DATA_PATH` | `data/opsa/safechain_train.parquet` | Training data path |
| Window size | `OPSA_WINDOW_SIZE` | `32` | Refusal-decision window (tokens) |
| Temperature | `OPSA_TEMPERATURE` | `1.0` | KL computation temperature |
| Decay type | `OPSA_DECAY_TYPE` | `linear` | Weight decay beyond window |
| Min weight | `OPSA_MIN_WEIGHT` | `0.1` | Minimum token weight outside window |
| Loss coef | `OPSA_LOSS_COEF` | `1.0` | OPSA distillation loss coefficient |
| GPUs | `GPUS_PER_NODE` | `8` | GPUs per node |
| Batch size | `TRAIN_BATCH_SIZE` | `128` | Global training batch size |

### Adjusting for Different Hardware

```bash
# 4x A100 setup
GPUS_PER_NODE=4 TRAIN_BATCH_SIZE=64 MINI_BATCH_SIZE=64 bash examples/opsa/run_opsa.sh

# Multi-node (2 nodes, 8 GPUs each)
NNODES=2 bash examples/opsa/run_opsa.sh
```

---

## Datasets

### SafeChain (Recommended)

```bash
python examples/opsa/prepare_safety_data.py --dataset UWNSL/SafeChain
```

40K samples of safe reasoning traces with harmful/benign labels.

### ThinkSafe (Per-Model)

```bash
# Choose the dataset matching your base model
python examples/opsa/prepare_safety_data.py --dataset Seanie-lee/ThinkSafe-Qwen3-0.6B
python examples/opsa/prepare_safety_data.py --dataset Seanie-lee/ThinkSafe-Qwen3-1.7B
python examples/opsa/prepare_safety_data.py --dataset Seanie-lee/ThinkSafe-DeepSeek-R1-Distill-Qwen-1.5B
```

---

## File Structure

```
examples/opsa/
├── run_opsa.sh              # Training launch script
├── prepare_safety_data.py   # Data download and preprocessing
└── README.md                # This file
```

---

## Troubleshooting

1. **OOM errors**: Reduce `TRAIN_BATCH_SIZE` or `MINI_BATCH_SIZE`, or use gradient checkpointing.
2. **Dataset not found**: Check HuggingFace access. Some datasets may require authentication: `huggingface-cli login`.
3. **Over-refusal**: Try reducing `OPSA_WINDOW_SIZE` or `OPSA_LOSS_COEF` to weaken the safety signal.
4. **Under-refusal**: Increase `OPSA_WINDOW_SIZE` or `OPSA_LOSS_COEF`, or use a higher quality safety dataset.

---

## Citation

```bibtex
@article{fu2026opsa,
  title={Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation},
  author={Fu, Yu and Yu, Longxuan and Shahgir, Haz Sameen and Wei, Zhipeng and Liu, Hui and Erichson, N. Benjamin and Dong, Yue},
  journal={arXiv preprint arXiv:2605.15239},
  year={2026}
}
```
