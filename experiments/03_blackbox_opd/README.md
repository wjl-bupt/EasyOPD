# 04_blackbox_opd — Black-box Online Policy Distillation

## Overview

This directory contains experiments for **black-box OPD** methods that do NOT require access to teacher model logits.

Instead of logit-level KD, these methods use rubric-based judging or API-only teacher signals, making them applicable when the teacher is a closed-source model (e.g., GPT-4, Claude).

## Methods

- **GRPO (Baseline)**: Standard GRPO reinforcement learning, no distillation
- **ROPD**: Rubric-based On-policy Distillation — uses rubric/judge pipeline to score student outputs against teacher-generated rubrics

## Results

| Method | Model | MATH-500 | GSM8K |
|--------|-------|----------|-------|
| GRPO (Baseline) | Qwen2.5-1.5B-Instruct | — | — |
| ROPD | Qwen2.5-1.5B-Instruct | — | — |

## Training Config

- **Student Model**: Qwen2.5-1.5B-Instruct (1.5B)
- **Teacher Model**: Black-box (API-only, no logit access)
- **Reward Signal**: Rubric-based judge (rubricator + verifier pipeline)
- **Hardware**: 8x NVIDIA H20 (96GB)

## Directory Structure

```
04_blackbox_opd/
├── data/
├── methods/
│   ├── grpo/
│   │   ├── checkpoints/
│   │   └── results/
│   └── ropd/
│       ├── checkpoints/
│       └── results/
└── README.md
```

## Notes

- ROPD does NOT require teacher logits — only text outputs or API access
- Uses a two-stage pipeline: rubricator generates scoring criteria, verifier scores student responses
- Applicable to closed-source teachers (GPT-4, Claude, etc.)
- Can be combined with any student model regardless of tokenizer compatibility
