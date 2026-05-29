# SDPO: Self-Distilled Policy Optimization

## Method Overview

**Paper:** [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) (Hübotter et al., 2026)

**Code:** [https://github.com/lasgroup/SDPO](https://github.com/lasgroup/SDPO)

SDPO (Self-Distilled Policy Optimization) augments on-policy reinforcement learning with self-distillation from the model's own high-reward trajectories. Unlike traditional knowledge distillation that requires an external teacher model, SDPO uses the **current model conditioned on feedback** as a self-teacher.

### Core Idea

1. **On-policy rollout**: The student model generates multiple responses per prompt.
2. **Identify successes**: Responses exceeding a reward threshold are marked as successful demonstrations.
3. **Reprompting**: For each sample, construct a "teacher prompt" that includes:
   - The original question
   - A successful demonstration from the same prompt group
   - (Optional) Environment feedback from failed attempts
4. **Self-distillation**: The model conditioned on this enriched prompt acts as a self-teacher. Its next-token predictions are distilled back into the policy using a generalized JSD loss.
5. **EMA teacher**: The teacher model weights are maintained as an EMA of the student.

### Key Innovation

SDPO converts tokenized feedback into a **dense learning signal** without any external teacher or explicit reward model. It leverages the model's ability to retrospectively identify its own mistakes in-context.

## Modified verl Files

| File | Modification | Reason |
|------|-------------|--------|
| `verl/trainer/ppo/core_algos.py` | Add `compute_self_distillation_loss` function | Core SDPO loss computation (JSD with IS correction) |
| `verl/workers/config/actor.py` | Add `SelfDistillationConfig` dataclass | Configuration for self-distillation parameters |
| `verl/workers/actor/dp_actor.py` | Add self-distillation forward pass in `update_policy` | Teacher forward + distillation loss in training loop |
| `verl/trainer/ppo/ray_trainer.py` | Add `_maybe_build_self_distillation_batch` method | Build teacher prompts from demonstrations/feedback |

## Configuration Parameters

### `actor.policy_loss`
- `loss_mode: "sdpo"` — Enables self-distillation mode

### `actor.self_distillation`
| Parameter | Default | Description |
|-----------|---------|-------------|
| `full_logit_distillation` | `True` | Use full-logit KL distillation |
| `alpha` | `0.5` | KL interpolation: 0.0=forward KL, 1.0=reverse KL, 0.5=JSD |
| `success_reward_threshold` | `1.0` | Minimum reward to be considered successful |
| `teacher_regularization` | `"ema"` | Teacher mode: "ema" or "trust-region" |
| `teacher_update_rate` | `0.05` | EMA update rate for teacher weights |
| `distillation_topk` | `100` | Top-k logits for distillation (None = full vocab) |
| `distillation_add_tail` | `True` | Add tail bucket for top-k distillation |
| `is_clip` | `2.0` | IS ratio clip value (None disables IS) |
| `max_reprompt_len` | `10240` | Maximum reprompted prompt length |
| `reprompt_truncation` | `"right"` | Truncation method for reprompted prompts |
| `dont_reprompt_on_self_success` | `True` | Don't use sample's own success as demonstration |
| `remove_thinking_from_demonstration` | `True` | Remove `<think>` tags from demonstrations |
| `include_environment_feedback` | `True` | Include environment feedback in reprompting |
| `environment_feedback_only_without_solution` | `True` | Only use feedback when no solution available |

## Reproduction Steps

1. **Data preparation:**
   ```bash
   python data/preprocess.py --data_source <DATASET_PATH>
   ```

2. **Run training:**
   ```bash
   bash examples/sdpo/run_sdpo.sh
   ```

3. **Key hyperparameters to tune:**
   - `alpha`: 0.5 (JSD) works well in most cases
   - `distillation_topk`: 100 (reduces memory, maintains quality)
   - `is_clip`: 2.0 (stabilizes training)
   - `teacher_update_rate`: 0.05 (slow EMA for stable teacher)

## Experimental Results

| Benchmark | GRPO (baseline) | SDPO |
|-----------|----------------|------|
| Chemistry (1h) | ~45% | ~55% |
| Chemistry (5h) | ~52% | ~60% |
| LiveCodeBench v6 | ~25% | ~32% |

*Results from the paper using Olmo3-7B-Instruct, avg@16.*

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    SDPO Training Loop                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Student generates N responses per prompt             │
│     ┌──────────┐                                        │
│     │ Student  │ ──→ [response_1, ..., response_N]      │
│     └──────────┘                                        │
│                                                          │
│  2. Reward function scores responses                     │
│     reward_i >= threshold → "successful demonstration"   │
│                                                          │
│  3. Build teacher prompt (reprompting)                   │
│     ┌─────────────────────────────────────────┐         │
│     │ Original Question                        │         │
│     │ + Successful Demonstration (if any)      │         │
│     │ + Environment Feedback (if available)    │         │
│     └─────────────────────────────────────────┘         │
│                                                          │
│  4. Self-teacher forward pass                            │
│     ┌──────────────┐                                    │
│     │ EMA Teacher  │ ──→ teacher_logits                 │
│     └──────────────┘                                    │
│                                                          │
│  5. Compute SDPO loss                                    │
│     L = JSD_alpha(student_logits, teacher_logits)        │
│       × IS_correction × self_distillation_mask           │
│                                                          │
│  6. Update student + EMA teacher                         │
│     θ_teacher ← (1-τ) × θ_teacher + τ × θ_student      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
