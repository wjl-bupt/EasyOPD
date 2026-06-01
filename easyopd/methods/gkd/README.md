# GKD: Generalized Knowledge Distillation (On-Policy Distillation)

## Method Overview

GKD (Generalized Knowledge Distillation) is an on-policy distillation method that
combines the advantages of reinforcement learning (on-policy sampling) and knowledge
distillation (dense teacher feedback). It addresses the train-inference distribution
mismatch problem in traditional KD by having the student generate sequences on-policy,
then using the teacher to provide per-token feedback via a Generalized Jensen-Shannon
Divergence (JSD).

**Paper:** [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
**Venue:** ICLR 2024

**Key Features:**
- **On-Policy Sampling**: Student generates sequences, avoiding distribution mismatch
- **Dense Teacher Feedback**: Per-token KL divergence (not sparse reward)
- **Generalized JSD**: β parameter interpolates between forward and reverse KL
- **Policy Gradient Integration**: Can use distillation loss as reward signal

---

## Algorithm: Generalized JSD

The core loss function is:

```
L_GKD = β * KL(student || teacher) + (1 - β) * KL(teacher || student)
```

### Key Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| **β (beta)** | [0, 1] | JSD interpolation: 0=forward KL (mean-seeking), 0.5=symmetric, 1=reverse KL (mode-seeking) |
| **λ (lambda)** | [0, 1] | On-policy ratio: 0=off-policy, 1=on-policy |
| **temperature** | > 0 | Softmax temperature for KL computation |

### Comparison with Other Methods

| Method | Sampling | Feedback | Distribution Match |
|--------|----------|----------|-------------------|
| SFT | Off-Policy | Dense | ❌ |
| RL (PPO/GRPO) | On-Policy | Sparse | ✅ |
| Traditional KD | Off-Policy | Dense | ❌ |
| **GKD** | **On-Policy** | **Dense** | **✅** |

---

## Integration with EasyOPD

GKD integrates naturally with verl's existing distillation framework:

1. **Loss Registration**: GKD's generalized JSD is registered as `loss_mode="gkd"` in verl's distillation loss registry
2. **On-Policy Support**: verl already supports `use_policy_gradient=True` which enables on-policy distillation
3. **Teacher Infrastructure**: verl's teacher model worker provides the dense per-token feedback

### Modified verl Files

| File | Modification | Reason |
|------|-------------|--------|
| `verl/trainer/distillation/losses.py` | Register `gkd`/`jsd`/`generalized_jsd` loss | Core GKD JSD loss computation |
| `verl/workers/config/distillation.py` | Add `gkd_beta`, `gkd_temperature` fields | GKD-specific configuration |

---

## Environment Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | >= 3.10 | |
| CUDA | >= 12.1 | |
| PyTorch | >= 2.4 | |
| vLLM | 0.8.x | For teacher inference |
| verl | From source | `pip install -e .` from EasyOPD root |

### Hardware Requirements

- **Student training**: 8 GPUs (1 node)
- **Teacher inference**: Additional GPUs via distillation resource pool
- Typical setup: 2 nodes (8 GPUs student + 8 GPUs teacher)

---

## Quick Start

### Step 1: Install EasyOPD

```bash
cd /path/to/EasyOPD
pip install -e .
pip install flash-attn --no-build-isolation
```

### Step 2: Prepare Data

Prepare training data in parquet format with a `content` field containing chat messages.

### Step 3: Run Training

```bash
bash examples/gkd/run_gkd.sh \
    --model Qwen/Qwen2-0.5B-Instruct \
    --teacher_model Qwen/Qwen2-1.5B-Instruct \
    --beta 0.5 \
    --nnodes 2
```

---

## Configuration Parameters

### Distillation Loss Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_mode` | "gkd" | Set to "gkd", "jsd", or "generalized_jsd" |
| `gkd_beta` | 0.5 | JSD interpolation (paper Eq.1: 0=forward KL, 1=reverse KL) |
| `gkd_temperature` | 1.0 | Softmax temperature |
| `use_policy_gradient` | True | On-policy mode (recommended for GKD) |
| `use_task_rewards` | True | Combine with task rewards |
| `distillation_loss_coef` | 1.0 | Weight of distillation loss |

### Recommended Configurations

#### Pure On-Policy GKD (Paper Default)
```bash
--loss_mode gkd --beta 0.5 --use_policy_gradient True
```

#### Supervised GKD (Direct Backprop)
```bash
--loss_mode gkd --beta 0.5 --use_policy_gradient False
```

#### Forward KL Only (Mean-Seeking, β=0)
```bash
--loss_mode gkd --beta 0.0
```

#### Reverse KL Only (Mode-Seeking, β=1)
```bash
--loss_mode gkd --beta 1.0
```

---

## File Structure

```
EasyOPD/
├── easyopd/methods/gkd/
│   ├── __init__.py              # Exports and GKDMethod metadata class
│   ├── core.py                  # Core algorithm (generalized JSD, on-policy ratio)
│   └── README.md                # This file
├── easyopd/config/gkd.yaml     # Config template
├── examples/gkd/
│   └── run_gkd.sh              # Training launch script
└── verl/
    ├── trainer/distillation/losses.py    # [EasyOPD:GKD] Register JSD loss
    └── workers/config/distillation.py    # [EasyOPD:GKD] Add gkd_beta, gkd_temperature
```

---

## Citation

```bibtex
@inproceedings{agarwal2024onpolicy,
  title={On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
  author={Agarwal, Rishabh and Vieillard, Nino and Zhou, Yongchao and Stanczyk, Piotr and Ramos, Sabela and Geist, Matthieu and Bachem, Olivier},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
