# 06_general_opd — General Online Policy Distillation

## Overview

This directory contains experiments for general-purpose OPD methods that don't fall into a specific specialized category (cross-tokenizer, agentic, blackbox, self-distillation, or vision).

These methods assume same-tokenizer teacher/student and use standard logit-level KD with various algorithmic improvements.

## Methods

- **GRPO (Baseline)**: Standard GRPO reinforcement learning, no distillation
- **GKD**: Generalized Knowledge Distillation — on-policy sampling + generalized JSD loss
- **G-OPD**: Generalized OPD with reward extrapolation + flexible reference model

## Results

| Method | Model | MATH-500 | GSM8K |
|--------|-------|----------|-------|
| GRPO (Baseline) | Qwen2.5-1.5B-Instruct | — | — |
| GKD | Qwen2.5-1.5B-Instruct | — | — |
| G-OPD | Qwen2.5-1.5B-Instruct | — | — |

## Training Config

- **Student Model**: Qwen2.5-1.5B-Instruct (1.5B)
- **Teacher Model**: Qwen2.5-7B-Instruct (7B)
- **Hardware**: 8x NVIDIA H20 (96GB)

## Directory Structure

```
06_general_opd/
├── data/
├── methods/
│   ├── grpo/
│   │   ├── checkpoints/
│   │   └── results/
│   ├── gkd/
│   │   ├── checkpoints/
│   │   └── results/
│   └── g_opd/
│       ├── checkpoints/
│       └── results/
└── README.md
```
