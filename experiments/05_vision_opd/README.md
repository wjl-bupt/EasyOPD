# 04_vision_opd — Vision Online Policy Distillation

## Overview

This directory contains experiments for multimodal (vision-language) online policy distillation.

The key difference from other categories: **datasets are multimodal** (image + text) and **benchmarks are different** (vision-language tasks instead of math-only). Results are not directly comparable with text-only experiments.

## Methods

- **GRPO (Baseline)**: Standard GRPO reinforcement learning for VLM, no distillation
- **Vision-OPD**: Multimodal self-distillation — EMA teacher with fine-grained visual inputs (bbox crops)

## Results

| Method | Model | MathVista | MathVerse |
|--------|-------|-----------|-----------|
| GRPO (Baseline) | Qwen2.5-VL-3B | — | — |
| Vision-OPD | Qwen2.5-VL-3B | — | — |

🚧 **Experiments pending** — results will be filled in after training completes.

## Training Config

- **Algorithm**: GRPO + EMA self-distillation with visual augmentation
- **Model**: Qwen2.5-VL-3B
- **Teacher**: Self (EMA with bbox-cropped fine-grained visual inputs)
- **Hardware**: 8x NVIDIA H20 (96GB)

## Directory Structure

```
04_vision_opd/
├── data/                        # Multimodal training data (image + text)
├── methods/
│   ├── grpo/                    # GRPO baseline (no distillation)
│   │   ├── checkpoints/
│   │   └── results/
│   └── vision_opd/             # Vision-OPD (multimodal EMA self-teacher)
│       ├── checkpoints/
│       └── results/
└── README.md
```

## Notes

- Vision-OPD uses bbox-cropped images for the EMA teacher to provide fine-grained visual guidance
- Datasets are multimodal (image + text), NOT shared with text-only experiments
- Benchmarks (MathVista, MathVerse) are different from text-only benchmarks (MATH-500, GSM8K)
