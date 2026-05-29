# G-OPD: Generalized On-Policy Distillation with Reward Extrapolation

## Method Overview

G-OPD introduces a reward scaling factor (λ) and a flexible reference model to generalize standard OPD.
Building on G-OPD, **ExOPD** (On-Policy Distillation with Reward Extrapolation) outperforms standard OPD
in both same-size and strong-to-weak distillation settings.

**Paper:** https://arxiv.org/abs/2602.12125
**Code:** https://github.com/RUCBM/G-OPD

**Key Features:**
- **Reward Scaling (λ):** Controls the strength of teacher signal (λ=1.0: OPD, λ>1.0: ExOPD)
- **Multi-Teacher Distillation:** Routes samples to domain-specific teachers (math/code)
- **Context Distillation:** Uses critique or reference solutions as teacher context
- **Rollout Correction:** Importance sampling + rejection sampling for off-policy correction

---

## Environment Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | >= 3.10 | Tested with 3.10 |
| CUDA | >= 12.1 | |
| PyTorch | >= 2.4 | |
| vLLM | 0.8.x | For rollout |
| verl | From source | `pip install -e .` from EasyOPD root |
| math-verify | latest | For math evaluation |

### Hardware Requirements

- **8x NVIDIA GPUs** (H100/A100/H20 recommended)
- batch_size=1024, n=1 -> 1024 rollout samples per step
- infer_tp=4
- Peak GPU memory depends on model size

---

## Quick Start

### Step 1: Install EasyOPD

```bash
cd /path/to/EasyOPD
pip install -e .
pip install math-verify
```

### Step 2: Configure Paths

Edit `examples/g_opd/run_g_opd.sh` and set:

```bash
STUDENT_MODEL_PATH="/path/to/student_model"    # e.g., Qwen/Qwen3-1.7B
TEACHER_MODEL_PATH="/path/to/teacher_model"    # e.g., Qwen/Qwen3-4B-Non-Thinking-RL-Math
BASE_MODEL_PATH="/path/to/base_model"          # Typically same as student initial state
TRAIN_DATA="/path/to/training_data.parquet"
VAL_DATA_1="/path/to/AIME2024/test.parquet"
VAL_DATA_2="/path/to/AIME2025/test.parquet"
```

### Step 3: Run Training

```bash
cd /path/to/EasyOPD
bash examples/g_opd/run_g_opd.sh
```

---

## File Structure

```
EasyOPD/
|-- easyopd/methods/g_opd/
|   |-- __init__.py              # Exports and GOPDMethod metadata class
|   |-- core.py                  # Core G-OPD algorithm (reward scaling, multi-teacher)
|   |-- ref_input_utils.py       # Context distillation utilities
|   +-- README.md                # This file
|-- easyopd/config/g_opd.yaml   # Config template (reference only)
|-- examples/g_opd/
|   |-- run_g_opd.sh            # Training launch script
|   +-- reward.py               # Math reward function
+-- verl/
    |-- workers/config/actor.py          # PolicyLossConfig: only_reverse_kl_advantages, lambda_vals, multi_teacher_distill
    |-- workers/actor/dp_actor.py        # G-OPD advantage computation branch
    |-- trainer/config/algorithm.py      # AlgoConfig: critique_vllm_url, use_ref_solution_distillation
    +-- trainer/ppo/ray_trainer.py       # Context distillation + base model log prob computation
```

---

## G-OPD Configuration Parameters

### Core Parameters (in `actor_rollout_ref.actor.policy_loss`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `only_reverse_kl_advantages` | False | Enable on-policy distillation (use reverse KL as advantages) |
| `lambda_vals` | 1.0 | Reward scaling factor (1.0=OPD, 1.25=ExOPD recommended) |
| `multi_teacher_distill` | False | Enable multi-teacher distillation |

### Context Distillation Parameters (in `algorithm`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `critique_vllm_url` | null | vLLM server URL for critique generation (enables context distillation) |
| `critique_model` | null | Model name on the critique server |
| `max_critique_tokens` | 2048 | Max tokens for critique generation |
| `critique_temperature` | 0.0 | Temperature for critique generation |
| `use_ref_solution_distillation` | False | Use ref_solution from dataset as teacher context |

### Base Model Parameters (for G-OPD/ExOPD reward normalization)

| Parameter | Description |
|-----------|-------------|
| `+actor_rollout_ref.model.base_model_path` | Base model for actor (typically student's initial state) |
| `+actor_rollout_ref.ref.model.base_model_path` | Base model for ref (same as actor base in single-teacher) |

---

## Usage Modes

### 1. Standard OPD (λ=1.0)

```bash
actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True
# No base_model_path needed
```

### 2. ExOPD (λ=1.25, recommended)

```bash
actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True
actor_rollout_ref.actor.policy_loss.lambda_vals=1.25
+actor_rollout_ref.model.base_model_path=Qwen/Qwen3-1.7B
+actor_rollout_ref.ref.model.base_model_path=Qwen/Qwen3-1.7B
```

### 3. Multi-Teacher Distillation

```bash
actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True
actor_rollout_ref.actor.policy_loss.lambda_vals=1.25
actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true
+actor_rollout_ref.ref.model.path=<math_teacher>
+actor_rollout_ref.ref.model.base_model_path=<code_teacher>
```

Data must include `extra_info.opd_teacher` field ("math" or "code") per sample.

### 4. On-Policy Context Distillation (OPSD)

```bash
algorithm.use_ref_solution_distillation=true
data.return_raw_chat=True
```

Data must include `extra_info.ref_solution` field per sample.

---

## Training Data Format

Training data should be in parquet format with fields:
- `prompt`: The input prompt/question
- `extra_info` (optional): Dict with additional fields:
  - `opd_teacher`: "math" or "code" (for multi-teacher mode)
  - `ref_solution`: Reference solution text (for context distillation mode)

Recommended datasets:
- [G-OPD Training Data](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data)
- DeepMath-103K (filtered level 6)

---

## Citation

```bibtex
@article{yang2026learning,
  title={Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation},
  author={Yang, Wenkai and Liu, Weijie and Xie, Ruobing and Yang, Kai and Yang, Saiyong and Lin, Yankai},
  journal={arXiv preprint arXiv:2602.12125},
  year={2026}
}
```
