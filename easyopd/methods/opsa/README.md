# OPSA: Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation

## Method Overview

OPSA (On-Policy Self-distillation for safety Alignment) addresses the "safety tax" — the well-known tradeoff where safety alignment improves robustness to harmful queries but reduces general reasoning ability. Prior work attributed this to distributional mismatch (off-policy SFT on fixed teacher-generated data), but OPSA identifies a second source: **off-policy learning itself**, where supervision is applied to fixed demonstrations rather than the model's own trajectories.

**Paper:** [Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation](https://arxiv.org/abs/2605.15239)  
**Authors:** Yu Fu, Longxuan Yu, Haz Sameen Shahgir, Zhipeng Wei, Hui Liu, N. Benjamin Erichson, Yue Dong  
**Code:** [https://github.com/FYYFU/OPSA](https://github.com/FYYFU/OPSA)

### Key Features

- **On-Policy Self-Distillation**: Uses the same base model as both teacher and student, eliminating the need for a separate (larger) teacher model
- **Type-Conditional Privileged Contexts**: Different system prompts for harmful queries (activates refusal) vs benign queries (suppresses over-refusal)
- **Safety-Critical Token Window**: Concentrates gradient updates on the narrow early "refusal-decision window" where safety behavior is determined
- **Teacher Flip Rate (TFR)**: A training-free signal to evaluate and select the most effective privileged contexts

---

## Algorithm: On-Policy Self-Distillation with Privileged Contexts

### Core Mechanism

1. **Student rollout**: The student policy generates on-policy responses to prompts (without privileged context)
2. **Teacher scoring**: A frozen copy of the same model, provided with a type-conditional privileged context, computes per-token logits
3. **KL supervision**: Per-token forward KL divergence D_KL(p_Teacher || p_Student) is applied, weighted by the early-window function
4. **Context selection**: Teacher Flip Rate (TFR) guides the search for effective privileged contexts

### Mathematical Formulation

**OPSA Loss (Eq. 3 from paper):**

```
L_OPSA = sum_t w(t) * D_KL(p_T(·|x, I, y_{<t}) || p_S(·|x, y_{<t}))
```

where:
- `p_T` = teacher distribution (with privileged context I)
- `p_S` = student distribution (without context)
- `w(t)` = early-window weight function
- `I` = type-conditional privileged context (I_h for harmful, I_b for benign)

**Teacher Flip Rate (Section 3.3):**

```
TFR(I) = |{x : unsafe(T(x)) AND safe(T(x|I))}| / |{x : unsafe(T(x))}|
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Softmax temperature for KL computation |
| `window_size` | 32 | Refusal-decision window size (tokens) |
| `decay_type` | "linear" | Weight decay beyond window: linear/step/exponential |
| `min_weight` | 0.1 | Minimum weight for non-window tokens |
| `loss_agg_mode` | "token-mean" | Loss aggregation: token-mean/seq-mean-token-sum/seq-mean-token-mean |
| `tfr_threshold` | 0.8 | Minimum TFR for context acceptance |

---

## Integration with EasyOPD

OPSA follows **Mode A** (lightweight verl modification):

- Core algorithm self-contained in `easyopd/methods/opsa/core.py`
- Minimal verl changes for config fields and training loop hooks
- Uses verl's existing on-policy rollout infrastructure

### Modified verl files

| File | Modification |
|------|-------------|
| `verl/workers/config/actor.py` | Added `OPSAConfig` dataclass for OPSA hyperparameters |
| `verl/workers/actor/dp_actor.py` | Added OPSA per-token KL loss computation with privileged contexts |
| `verl/trainer/ppo/ray_trainer.py` | Added privileged context injection logic during rollout |

---

## Environment Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | >= 3.10 | |
| CUDA | >= 12.1 | For GPU training |
| PyTorch | >= 2.4 | |
| vLLM | >= 0.6.3 | For rollout generation |
| transformers | >= 4.45 | Model loading |

### Hardware Requirements

- Minimum: 4x A100 (80GB) or equivalent
- Recommended: 8x A100 (80GB) for full-scale training
- Models tested: Qwen3-0.6B, Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B

---

## Quick Start

### Step 1: Install EasyOPD

```bash
pip install -e .
```

### Step 2: Prepare Safety Data

```bash
python examples/opsa/prepare_safety_data.py \
    --dataset UWNSL/SafeChain \
    --output_dir data/opsa/
```

### Step 3: Configure Paths

Edit `easyopd/config/opsa.yaml` to set your model and data paths.

### Step 4: Run Training

```bash
bash examples/opsa/run_opsa.sh
```

---

## File Structure

```
easyopd/methods/opsa/
├── __init__.py           # Method metadata (OPSAMethod) + exports
├── core.py               # Core algorithm: TFR, window weights, KL loss, opsa_loss
└── README.md             # This file

easyopd/config/
└── opsa.yaml             # OPSA configuration template

examples/opsa/
├── run_opsa.sh           # Training launch script
├── prepare_safety_data.py # Data preparation utility
└── README.md             # Usage guide
```

---

## Configuration Parameters

### OPSA-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `opsa.temperature` | float | 1.0 | KL computation temperature |
| `opsa.window_size` | int | 32 | Early refusal-decision window size |
| `opsa.decay_type` | str | "linear" | Weight decay type beyond window |
| `opsa.min_weight` | float | 0.1 | Minimum token weight outside window |
| `opsa.use_window_weighting` | bool | true | Enable/disable window weighting |
| `opsa.loss_agg_mode` | str | "token-mean" | Loss aggregation method |
| `opsa.distillation_loss_coef` | float | 1.0 | OPSA loss coefficient |

### TFR Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `opsa.tfr_threshold` | float | 0.8 | Minimum TFR for context acceptance |
| `opsa.tfr_sample_size` | int | 100 | Number of samples for TFR evaluation |

### Privileged Context Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `opsa.harmful_context` | str | (see config) | Context for harmful query handling |
| `opsa.benign_context` | str | (see config) | Context for benign query handling |
| `opsa.context_search_method` | str | "tfr" | Context selection method |

---

## Datasets

OPSA uses safety-specific datasets:

- **SafeChain** (`UWNSL/SafeChain`): 40K samples of safe reasoning traces
- **ThinkSafe** (`Seanie-lee/ThinkSafe-*`): Per-model safety datasets

---

## Validated Results

Results from the original paper on safety benchmarks:

| Model | HarmBench ↓ | XSTest ↓ | WildJailbreak ↓ | MATH (reasoning) |
|-------|------------|----------|-----------------|-------------------|
| Qwen3-1.7B (base) | High | High | High | Baseline |
| Qwen3-1.7B + SFT | Low | Medium | Medium | Degraded |
| Qwen3-1.7B + OPSA | Low | Low | Low | Preserved |

---

## Troubleshooting

### Common Issues

1. **OOM during self-distillation**: The frozen teacher copy doubles memory usage. Use `gradient_checkpointing=True` or reduce batch size.
2. **TFR too low**: The privileged context may not be effective for your model. Try different context templates or lower the `tfr_threshold`.
3. **Over-refusal on benign queries**: Increase the proportion of benign data in `label_ratios` or adjust the benign context template.

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
