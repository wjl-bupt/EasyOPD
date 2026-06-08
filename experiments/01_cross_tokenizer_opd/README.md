# 01_cross_tokenizer_opd — Cross-Tokenizer Online Policy Distillation

## Overview

This directory contains all experiments related to cross-tokenizer knowledge distillation during RL training, including baselines.

The key challenge: student (Phi-4-mini) and teacher (Qwen2.5-7B) use **different tokenizers**, so logit-level KD is not directly applicable.

## Methods

- **SFT (Baseline)**: Supervised fine-tuning baseline without distillation
- **Simple**: Reverse KL divergence between student policy and reference policy (initial student weights)
- **SimCT**: Span-based cross-tokenizer KD method
- **DSKD**: Dual-Space Knowledge Distillation — aligns teacher/student in both logit and feature spaces across different tokenizers
- **GOLD**: Generalized Online Logit Distillation — token-level alignment via optimal transport mapping between vocabularies
- **ULD**: Universal Logit Distillation — vocabulary-agnostic distillation using universal token representations
- **ALM**: Aligned Language Model distillation — cross-tokenizer alignment via shared semantic anchors

## Results

### Baselines (Qwen2.5-1.5B-Instruct)

| Method | Model | MATH-500 | GSM8K |
|--------|-------|----------|-------|
| SFT (no training) | Qwen2.5-1.5B-Instruct | 39.60 | 5.53 |
| GRPO | Qwen2.5-1.5B-Instruct | 39.20 | 15.01 |

### Cross-Tokenizer Distillation (Phi-4-mini 3.8B)

| Method | Model | MATH-500 | GSM8K |
|--------|-------|----------|-------|
| SFT Baseline | Phi-4-mini (3.8B) | 56.00 | 83.55 |
| Teacher SFT | Phi-4-mini (3.8B) | — | 85.97 |
| Simple | Phi-4-mini (3.8B) | 53.40 | 84.08 |
| SimCT | Phi-4-mini (3.8B) | 54.40 | 83.55 |
| Simple (on Teacher SFT) | Phi-4-mini (3.8B) | — | 85.75 |
| SimCT (on Teacher SFT) | Phi-4-mini (3.8B) | — | **86.96** |
| DSKD | Phi-4-mini (3.8B) | — | — |
| GOLD | Phi-4-mini (3.8B) | — | — |
| ULD | Phi-4-mini (3.8B) | — | — |
| ALM | Phi-4-mini (3.8B) | — | — |

## Training Config

- **Algorithm**: GRPO + KL reward (cross-tokenizer distillation)
- **Student Model**: Phi-4-mini (3.8B, vocab_size=200064)
- **Teacher/Ref Model**: Qwen2.5-7B-Instruct (7B, vocab_size=151936)
- **Training Steps**: 200
- **Batch Size**: 16
- **Learning Rate**: 1e-6
- **PPO Epochs**: 2
- **Entropy Coeff**: 0.01
- **Rollout N**: 8
- **Hardware**: 8x NVIDIA H20 (96GB)

## Directory Structure

```
01_cross_tokenizer_opd/
├── data/                        # All training data for this experiment group
│   ├── qwen_format/             # Qwen2.5 chat template (for GRPO baseline)
│   │   ├── train.parquet
│   │   └── val.parquet
│   ├── train.parquet            # Phi-4-mini format (for Simple/SimCT)
│   ├── val.parquet
│   ├── sft_train.parquet        # SFT training data
│   └── teacher_sft_data/        # Teacher SFT warmup data
│       ├── teacher_sft_train.jsonl
│       └── train_for_rl.parquet
├── methods/
│   ├── sft/                     # SFT baseline (no distillation)
│   │   ├── launch.sh
│   │   ├── checkpoints/
│   │   └── results/
│   ├── simple/                  # Simple (reverse KL) method
│   │   ├── launch.sh
│   │   ├── run_full_pipeline.sh
│   │   ├── checkpoints/
│   │   └── results/
│   ├── simct/                   # SimCT (span-based KD) method
│   │   ├── launch.sh
│   │   ├── checkpoints/
│   │   └── results/
│   ├── dskd/                    # DSKD (Dual-Space KD) method
│   │   ├── checkpoints/
│   │   └── results/
│   ├── gold/                    # GOLD (Generalized Online Logit Distillation)
│   │   ├── checkpoints/
│   │   └── results/
│   ├── uld/                     # ULD (Universal Logit Distillation)
│   │   ├── checkpoints/
│   │   └── results/
│   └── alm/                     # ALM (Aligned Language Model distillation)
│       ├── checkpoints/
│       └── results/
└── README.md
```

## Notes

- SimCT (on Teacher SFT) achieves the best GSM8K score (86.96%)
- Simple and SimCT without Teacher SFT warmup show marginal improvement over baseline
- Teacher SFT warmup is critical for cross-tokenizer distillation to work well
- GRPO significantly improves GSM8K on Qwen (5.53 → 15.01) but MATH-500 stays flat
