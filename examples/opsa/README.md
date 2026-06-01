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

数据准备脚本会下载并处理安全数据集，输出**单个 parquet 文件**用于训练。
验证集默认复用训练集所在文件（即与训练集相同），无需单独切分。

```bash
# Download and process SafeChain dataset (outputs a single parquet file)
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
| Data | `DATA_PATH` | `data/opsa/safechain_train.parquet` | Training data path (验证集默认与训练集相同) |
| Window size | `OPSA_WINDOW_SIZE` | `32` | Refusal-decision window (tokens) |
| Temperature | `OPSA_TEMPERATURE` | `1.0` | KL computation temperature |
| Decay type | `OPSA_DECAY_TYPE` | `linear` | Weight decay beyond window |
| Min weight | `OPSA_MIN_WEIGHT` | `0.1` | Minimum token weight outside window |
| Loss coef | `OPSA_LOSS_COEF` | `1.0` | OPSA distillation loss coefficient |
| GPUs | `GPUS_PER_NODE` | `8` | GPUs per node |
| Batch size | `TRAIN_BATCH_SIZE` | `128` | Global training batch size |

> 说明：当前实现不再使用独立的 `VAL_DATA_PATH` 与 `--val_ratio` 切分参数；
> 验证阶段默认复用 `DATA_PATH` 指定的训练集文件。

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

## Evaluation

训练完成后，可使用 `examples/opsa/evaluation/` 下的脚本对训练得到的 checkpoint
进行多基准安全评估，量化 OPSA 在 refusal 与 over-refusal 两个维度的表现。

### 支持的安全基准

| Benchmark | 类型 | 评估目标 |
|-----------|------|----------|
| **HarmBench** | Harmful | 模型对有害指令的拒绝率（越高越安全） |
| **XSTest** | Mixed | 安全/不安全提示的区分能力（over-refusal 检测） |
| **WildJailbreak** | Adversarial | 模型对越狱攻击的鲁棒性 |
| **StrongReject** | Harmful | 严苛有害指令下的拒绝质量 |
| **WildBenign** | Benign | 良性请求的过度拒绝率（越低越好） |

### 运行评估

```bash
# 使用默认 guard model 评估训练好的 checkpoint
MODEL_PATH=outputs/opsa/checkpoint bash examples/opsa/evaluation/run_eval.sh
```

### Guard Model 依赖

安全评估需要一个 **guard model** 用于自动判定回复是否安全，目前支持以下两种：

- **WildGuard** (`allenai/wildguard`) — 默认推荐
- **LlamaGuard** (`meta-llama/Meta-Llama-Guard-2-8B`) — 备选

可通过环境变量切换：

```bash
GUARD_MODEL=allenai/wildguard \
MODEL_PATH=outputs/opsa/checkpoint \
bash examples/opsa/evaluation/run_eval.sh
```

更多评估细节（每个基准的指标定义、guard model prompt、batch size 调优等）
请参考 [`examples/opsa/evaluation/README.md`](evaluation/README.md)。

---

## File Structure

```
examples/opsa/
├── run_opsa.sh              # Training launch script
├── prepare_safety_data.py   # Data download and preprocessing
├── evaluation/              # Safety evaluation scripts
│   ├── run_eval.sh          # Evaluation launch script
│   ├── run_safety_eval.py   # Multi-benchmark evaluator
│   ├── guard_model.py       # WildGuard / LlamaGuard wrapper
│   └── README.md            # Evaluation details
└── README.md                # This file
```

---

## Troubleshooting

1. **OOM errors**: Reduce `TRAIN_BATCH_SIZE` or `MINI_BATCH_SIZE`, or use gradient checkpointing.
2. **Dataset not found**: Check HuggingFace access. Some datasets may require authentication: `huggingface-cli login`.
3. **Over-refusal**: Try reducing `OPSA_WINDOW_SIZE` or `OPSA_LOSS_COEF` to weaken the safety signal.
4. **Under-refusal**: Increase `OPSA_WINDOW_SIZE` or `OPSA_LOSS_COEF`, or use a higher quality safety dataset.
5. **Evaluation guard model OOM**: 评估时 guard model 与被评估模型同时驻留显存，可通过
   `GUARD_BATCH_SIZE` 降低批次，或将 guard model 部署在独立 GPU 上。
6. **Evaluation benchmark download failed**: 部分基准（如 HarmBench、WildJailbreak）
   需要 HuggingFace 登录，请先执行 `huggingface-cli login`。
7. **WildBenign 过度拒绝率偏高**: 说明安全信号过强，可适当下调 `OPSA_LOSS_COEF`
   或缩小 `OPSA_WINDOW_SIZE` 后重新训练。

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
