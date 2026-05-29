# OPCD: On-Policy Context Distillation for Language Models

## Method Overview

OPCD (On-Policy Context Distillation) distills contextual knowledge (experiential
knowledge, system prompts) into a language model using on-policy KL divergence
minimization. The teacher model receives context (experience) in its prompt while
the student generates responses without the context, then the student is trained
to match the teacher's output distribution via KL loss.

**Paper:** https://arxiv.org/abs/2602.12275
**Code:** https://github.com/microsoft/LMOps/tree/main/opcd

**Key Features:**
- **Two-Stage Pipeline**: Experience extraction → Knowledge consolidation
- **On-Policy KL Distillation**: Student generates, teacher provides logits with context
- **Multiple KL Types**: Full KL, forward KL, reverse KL, MSE, SeqKD, low-variance KL
- **Top-K Memory Optimization**: Use top-k logits for full KL to save GPU memory
- **Experience Injection**: Flexible experience injection into teacher prompts

---

## Environment Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | >= 3.10 | Tested with 3.11 |
| CUDA | >= 12.1 | |
| PyTorch | >= 2.4 | |
| vLLM | 0.8.x | For rollout |
| verl | From source | `pip install -e .` from EasyOPD root |
| flash-attn | latest | For efficient attention |

### Hardware Requirements

- **16x NVIDIA GPUs** (2 nodes × 8 GPUs recommended)
- batch_size=128, max_response_length=16384
- Peak GPU memory depends on model size and context length

---

## Quick Start

### Step 1: Install EasyOPD

```bash
cd /path/to/EasyOPD
pip install -e .
pip install flash-attn --no-build-isolation
```

### Step 2: Prepare Experience Data

OPCD requires a pre-extracted experience file (JSON format):

```json
[
    {"prompt": "Solve x^2 + 2x + 1 = 0", "experience": "This is a perfect square trinomial..."},
    {"prompt": "Find the derivative of sin(x)", "experience": "Use the chain rule..."}
]
```

Or as a dictionary:
```json
{
    "Solve x^2 + 2x + 1 = 0": "This is a perfect square trinomial...",
    "Find the derivative of sin(x)": "Use the chain rule..."
}
```

### Step 3: Configure Paths

Edit `examples/opcd/run_opcd_math.sh` and set:

```bash
MODEL_PATH="Qwen/Qwen3-8B"
EXP_PATH="/path/to/experience.json"
```

### Step 4: Run Training

```bash
cd /path/to/EasyOPD
bash examples/opcd/run_opcd_math.sh --model Qwen/Qwen3-8B --exp_path /path/to/experience.json
```

---

## File Structure

```
EasyOPD/
├── easyopd/methods/opcd/
│   ├── __init__.py              # Exports and OPCDMethod metadata class
│   ├── core.py                  # Core algorithm (KL penalty, OPCD loss, experience prompt)
│   └── README.md                # This file
├── easyopd/config/opcd.yaml     # Config template (reference only)
├── examples/opcd/
│   └── run_opcd_math.sh         # Training launch script for math consolidation
└── verl/
    ├── workers/config/actor.py          # Added kl_topk, kl_renorm_topk, profile_kl
    ├── workers/actor/dp_actor.py        # OPCD KL loss computation branch + select_keys
    └── trainer/ppo/ray_trainer.py       # _maybe_build_opcd_batch + consolidate stage logic
```

---

## OPCD Configuration Parameters

### Actor Config (in `actor_rollout_ref.actor`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_kl_loss` | False | Set to True to enable KL loss |
| `kl_loss_type` | "low_var_kl" | KL type: "full", "kl", "abs", "mse", "low_var_kl", "seqkd" |
| `kl_topk` | 0 | Top-k logits for full KL (0 = all vocab) |
| `kl_renorm_topk` | False | Renormalize top-k log-probs before KL |

### Trainer Config (in `trainer`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage` | None | Set to "consolidate" for OPCD consolidation |
| `experience_path` | None | Path to experience JSON file |
| `on_policy_merge` | True | On-policy (student generates) vs off-policy (teacher generates) |
| `experience_max_length` | 16384 | Max token length for experience text |
| `train_system_prompt` | False | Use system prompt mode for experience injection |

---

## KL Loss Types

| Type | Formula | Use Case |
|------|---------|----------|
| `full` | Σ p_i * (log p_i - log q_i) | Full vocabulary KL (recommended) |
| `kl` | log p - log q | Token-level forward KL |
| `abs` | \|log p - log q\| | Absolute difference |
| `mse` | 0.5 * (log p - log q)² | Mean squared error |
| `low_var_kl` | exp(log q - log p) - (log q - log p) - 1 | Low-variance KL estimator |
| `seqkd` | -log q | Sequence-level KD (cross-entropy with teacher) |

---

## Usage Modes

### 1. Math Consolidation (Recommended)

```bash
bash examples/opcd/run_opcd_math.sh \
    --model Qwen/Qwen3-8B \
    --exp_path /path/to/math_experience.json \
    --kl_loss_type full \
    --kl_topk 256 \
    --nnodes 2
```

### 2. System Prompt Distillation

```bash
bash examples/opcd/run_opcd_math.sh \
    --model Qwen/Qwen3-8B \
    --exp_path /path/to/system_prompts.json \
    --kl_loss_type full \
    --kl_topk 256
```

### 3. Custom KL Type

```bash
bash examples/opcd/run_opcd_math.sh \
    --model Qwen/Qwen3-8B \
    --exp_path /path/to/experience.json \
    --kl_loss_type low_var_kl
```

---

## Algorithm Details

### Two-Stage Pipeline

1. **Experience Extraction** (offline):
   - Run the model on training problems with chain-of-thought
   - Extract successful reasoning patterns as "experience"
   - Save to JSON file

2. **Knowledge Consolidation** (this stage):
   - Student generates responses WITHOUT experience context
   - Teacher (ref model) receives experience in its prompt
   - Student is trained to match teacher's output distribution via KL loss
   - Result: Student internalizes the experience knowledge

### On-Policy vs Off-Policy

- **On-policy** (`on_policy_merge=True`): Student generates responses, teacher provides logits
  - KL direction: KL(student || teacher) — student learns to match teacher
  - More stable, recommended

- **Off-policy** (`on_policy_merge=False`): Teacher generates responses
  - KL direction: KL(teacher || student) — student learns from teacher's generations
  - Can be more sample-efficient but less stable

---

## Citation

```bibtex
@article{cheng2025opcd,
  title={On-Policy Context Distillation for Language Models},
  author={Cheng, Hao and Liu, Xiaodong and Perot, Vincent and Gao, Jianfeng and Meek, Christopher},
  journal={arXiv preprint arXiv:2602.12275},
  year={2025}
}
```
