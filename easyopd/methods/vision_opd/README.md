# Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs via On-Policy Self-Distillation

## Method Overview

Vision-OPD uses **on-policy self-distillation** with a teacher model (maintained via EMA of the student)
that receives fine-grained visual inputs (e.g., bounding-box cropped images) to guide the student model
which only sees the original image. This enables the student to learn fine-grained visual perception
without requiring additional annotations at inference time.

**Paper:** https://arxiv.org/abs/2605.18740
**Code:** https://github.com/VisionOPD/Vision-OPD

**Key Features:**
- **Self-Distillation with EMA Teacher**: Teacher model is an EMA copy of the student, updated after each step
- **Fine-Grained Visual Inputs**: Teacher receives bbox-cropped images for enhanced perception
- **Top-K Logit Distillation**: Memory-efficient distillation using top-k logits with tail bucket
- **Generalized JSD Loss**: Supports forward KL, reverse KL, and Jensen-Shannon Divergence (alpha=0.5)
- **IS Ratio Clipping**: Importance sampling ratio clipping for training stability
- **Rollout Correction**: Token-level IS weights to handle off-policy issues
- **OPSD Answer Hint Mode**: Alternative mode using ground truth as teacher hint

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
| flashinfer | 0.6.6 | Optional, for inference |

### Hardware Requirements

- **8x NVIDIA GPUs** (H100/A100 recommended)
- batch_size=96, rollout_n=8 -> 768 rollout samples per step
- Peak GPU memory depends on model size and max_model_len

---

## Quick Start

### Step 1: Install EasyOPD

```bash
cd /path/to/EasyOPD
pip install -e .
pip install flash-attn --no-build-isolation
```

### Step 2: Prepare Training Data

Download the [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) dataset:

```bash
python scripts/prepare_data.py --data-dir ./data
```

The dataset should contain:
- `images`: Original images for the student
- `bbox_images`: Bounding-box cropped images for the teacher
- Standard prompt/response fields

### Step 3: Configure Paths

Edit `examples/vision_opd/run_vision_opd.sh` and set:

```bash
MODEL_PATH="Qwen/Qwen3.5-4B"
TASK_TRAIN_FILE="/path/to/train.parquet"
```

### Step 4: Run Training

```bash
cd /path/to/EasyOPD
bash examples/vision_opd/run_vision_opd.sh
```

---

## File Structure

```
EasyOPD/
|-- easyopd/methods/vision_opd/
|   |-- __init__.py              # Exports and VisionOPDMethod metadata class
|   |-- core.py                  # Core algorithm (self-distillation loss, EMA update, top-k utils)
|   |-- teacher_utils.py         # Teacher input preparation (image swap, answer hint)
|   +-- README.md                # This file
|-- easyopd/config/vision_opd.yaml  # Config template (reference only)
|-- examples/vision_opd/
|   +-- run_vision_opd.sh        # Training launch script
+-- verl/
    |-- workers/config/actor.py          # SelfDistillationConfig, 'vopd' loss mode
    |-- workers/actor/dp_actor.py        # VOPD loss computation + teacher EMA update
    |-- trainer/config/algorithm.py      # RolloutCorrectionConfig
    +-- trainer/ppo/ray_trainer.py       # _maybe_build_vision_opd_batch
```

---

## Vision-OPD Configuration Parameters

### Core Parameters (in `actor_rollout_ref.actor.policy_loss`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_mode` | "vanilla" | Set to "vopd" to enable Vision-OPD self-distillation |

### Self-Distillation Parameters (in `actor_rollout_ref.actor.self_distillation`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `full_logit_distillation` | True | Use full-logit KL distillation |
| `alpha` | 0.0 | KL interpolation: 0.0=forward KL, 1.0=reverse KL, 0.5=JSD |
| `gamma` | 1.0 | Weight for the distillation loss |
| `distillation_topk` | None | Top-k logits for memory-efficient distillation |
| `distillation_add_tail` | True | Add tail probability bucket for top-k |
| `is_clip` | None | IS ratio clip value (e.g., 2.0) |
| `teacher_regularization` | "ema" | Teacher update mode: "ema", "trust-region", "progressive" |
| `teacher_update_rate` | 0.05 | EMA update rate |
| `teacher_update_interval` | None | Hard-sync interval for progressive mode |
| `teacher_always_on` | False | Distill every sample from teacher |
| `teacher_model_source` | "legacy" | Teacher source: "legacy" (EMA), "current", "fixed" |
| `teacher_image_key` | None | Dataset column for teacher images (e.g., "bbox_images") |
| `teacher_prompt_mode` | None | Set to "answer_hint" for OPSD mode |
| `max_reprompt_len` | 10240 | Max teacher prompt length |
| `dont_reprompt_on_self_success` | False | Skip reprompting on successful samples |
| `fallback_to_policy_loss_on_missing_teacher` | False | Fall back to GRPO when teacher unavailable |

### Rollout Correction Parameters (in `algorithm.rollout_correction`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_is` | "sequence" | IS weight level: "token" or "sequence" |
| `rollout_is_threshold` | 2.0 | Upper threshold for IS weight truncation |

---

## Usage Modes

### 1. Vision-OPD with BBox Images (Recommended)

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=vopd
actor_rollout_ref.actor.self_distillation.teacher_always_on=True
actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images
actor_rollout_ref.actor.self_distillation.alpha=0.5
actor_rollout_ref.actor.self_distillation.distillation_topk=100
actor_rollout_ref.actor.self_distillation.is_clip=2.0
actor_rollout_ref.actor.self_distillation.teacher_regularization=ema
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
```

### 2. OPSD with Answer Hint

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=vopd
actor_rollout_ref.actor.self_distillation.teacher_always_on=True
actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=answer_hint
actor_rollout_ref.actor.self_distillation.alpha=0.5
```

### 3. Standard SDPO (Self-Distillation from Successful Rollouts)

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=vopd
actor_rollout_ref.actor.self_distillation.teacher_always_on=False
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True
```

---

## Training Data Format

Training data should be in parquet format with fields:
- `prompt`: The input prompt/question (with image placeholders)
- `images`: List of original images for the student
- `bbox_images`: List of bounding-box cropped images for the teacher
- `reward_model.ground_truth`: Ground truth answer (for OPSD mode)

Recommended dataset:
- [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K)

---

## Citation

```bibtex
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```
