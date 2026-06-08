# 03_self_opd — Self-Distillation Online Policy Distillation

## Overview

This directory contains experiments for self-distillation methods that do **not** require an external teacher model.

The "teacher" signal comes from the model itself — either via EMA weights, context-conditioned prompting, or high-reward trajectory replay. Only **one model** is needed.

## Methods

- **GRPO (Baseline)**: Standard GRPO reinforcement learning with math reward, no distillation
- **SDPO**: Self-Distilled Policy Optimization — EMA teacher + reprompting with successful demonstrations
- **OPCD**: On-Policy Context Distillation — self-teacher with experience/context injection

## Results

| Method | Model | MATH-500 | GSM8K |
|--------|-------|----------|-------|
| GRPO (Baseline) | Qwen2.5-1.5B-Instruct | — | — |
| SDPO | Qwen2.5-1.5B-Instruct | — | — |
| OPCD | Qwen2.5-1.5B-Instruct | — | — |

🚧 **Experiments pending** — results will be filled in after training completes.

## Training Config

- **Algorithm**: GRPO + self-distillation (no external teacher)
- **Model**: Qwen2.5-1.5B-Instruct
- **Teacher**: Self (EMA / context-conditioned)
- **Hardware**: 8x NVIDIA H20 (96GB)

## Directory Structure

```
03_self_opd/
├── data/                        # Training data
├── methods/
│   ├── grpo/                    # GRPO baseline (no distillation)
│   │   ├── checkpoints/
│   │   └── results/
│   ├── sdpo/                    # SDPO (EMA self-teacher + reprompting)
│   │   ├── checkpoints/
│   │   └── results/
│   └── opcd/                    # OPCD (context self-teacher)
│       ├── checkpoints/
│       └── results/
└── README.md
```

## Notes

- SDPO uses the model's own high-reward trajectories as demonstrations for self-teaching
- OPCD injects "experience" context into the teacher prompt to create a stronger self-teacher
- All methods in this group require NO external teacher model, reducing compute requirements
