# 03_agentic_opd — Agentic Online Policy Distillation

## Overview

This directory contains experiments for OPD methods specifically designed for **agent/tool-integrated reasoning (TIR)** scenarios.

The key challenge: multi-turn agent interactions with tool calls introduce cascade failures and step-wise divergence, requiring adaptive distillation strategies.

## Methods

- **GRPO (Baseline)**: Standard GRPO reinforcement learning for agent tasks, no distillation
- **SOD**: Step-wise On-policy Distillation — adaptive re-weighting based on per-step divergence trajectory

## Results

| Method | Model | AIME 2024 | AIME 2025 | LiveCodeBench |
|--------|-------|-----------|-----------|---------------|
| GRPO (Baseline) | Qwen3-1.7B | — | — | — |
| SOD | Qwen3-1.7B | — | — | — |

## Training Config

- **Student Model**: Qwen3-1.7B-SFT
- **Teacher Model**: DemyAgent-4B (Gen-Verse)
- **Dataset**: Open-AgentRL-30K (multi-turn agent interactions)
- **Hardware**: 8x NVIDIA H20 (96GB)

## Directory Structure

```
03_agentic_opd/
├── data/
├── methods/
│   ├── grpo/
│   │   ├── checkpoints/
│   │   └── results/
│   └── sod/
│       ├── checkpoints/
│       └── results/
└── README.md
```

## Notes

- SOD uses step-wise adaptive KL weighting to handle cascade failures in multi-turn agent reasoning
- Requires sandbox code execution service (SANDBOX_FUSION_URL)
- Benchmarks are agent-specific (AIME, LiveCodeBench), different from standard math benchmarks
