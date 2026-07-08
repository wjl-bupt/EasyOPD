# 04_self_opd — Self-Distillation Online Policy Distillation

## Overview

This directory contains experiments for self-distillation methods that do **not**
require an external teacher model. The "teacher" signal comes from the model
itself — via reprompting with successful demonstrations / feedback, EMA weights,
or context-conditioned prompting. Only **one model** is needed.

## Methods

- **GRPO (Baseline)**: Standard GRPO RL with math reward, no distillation.
- **SDPO** ✅ *implemented*: Self-Distillation Policy Optimization — the live
  policy conditioned on a correct demonstration / feedback from the rollout
  group acts as a self-teacher; the student is distilled towards it.
  - Paper: [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) (Hübotter et al., 2026)
  - Code:  https://github.com/lasgroup/SDPO
- **OPCD**: On-Policy Context Distillation — self-teacher with experience/context injection. *(planned)*

## SDPO at a glance

For each prompt, the policy samples a *group* of `rollout.n` attempts. The
verifier scores each. A failed attempt is **reprompted** with a correct
demonstration drawn from a *successful* attempt in the **same group** (and/or
textual environment feedback); the live policy then re-scores the failed
attempt's original response under that feedback-informed context — the
**self-teacher** `π(·|x, f)`. The loss distills the student `π(·|x)` towards the
stop-gradient self-teacher (paper Eq. 1):

```
L_SDPO = Σ_t KL( π(·|x, y_<t) ‖ stopgrad(π(·|x, f, y_<t)) )
```

with `alpha` interpolating forward/reverse KL (0.5 = symmetric JSD, the
paper-recommended stable choice). Samples without a usable self-teacher fall
back to standard GRPO. Combined as `gamma * L_SDPO + L_GRPO_fallback`.

Implementation:
- Algorithm: `easyopd/methods/sdpo/core.py` (`compute_sdpo_loss`,
  `build_sdpo_teacher_inputs`, `compute_sdpo_actor_loss`)
- Hooks: `easyopd/methods/sdpo/hooks.py` (`SDPOLossHook`, `SDPOTeacherSidecarHook`)
- verl touchpoints (`# [EasyOPD:SDPO]`): `verl/workers/actor/dp_actor.py`
  (loss branch + teacher field passthrough), `verl/trainer/ppo/ray_trainer.py`
  (`_maybe_build_sdpo_batch` reprompt construction).

## Running

```bash
# Single 8-GPU node (defaults: Qwen2.5-1.5B-Instruct, rollout.n=8, alpha=0.5)
bash experiments/04_self_opd/methods/sdpo/launch.sh

# Override common knobs via env vars:
STUDENT_MODEL=/path/to/model ROLLOUT_N=8 SDPO_ALPHA=0.5 SDPO_GAMMA=1.0 \
TOTAL_EPOCHS=2 \
bash experiments/04_self_opd/methods/sdpo/launch.sh

# Quick smoke run (override hydra directly after `--`):
bash experiments/04_self_opd/methods/sdpo/launch.sh \
    trainer.total_training_steps=5 trainer.save_freq=5
```

The script prepares the RL prompt parquet, (re)starts Ray, trains via
`verl.trainer.main_ppo` with `actor_rollout_ref.actor.policy_loss.loss_mode=sdpo`,
merges each FSDP checkpoint to HF format, and evaluates on MATH-500 + GSM8K.

> Note: `rollout.n` **must be > 1** so each prompt group contains both
> successful and failed attempts — that is what supplies the self-teacher's
> demonstration in scalar-reward (RLVR) environments.

## Results

| Method | Model | MATH-500 | GSM8K |
|--------|-------|----------|-------|
| GRPO (Baseline) | Qwen2.5-1.5B-Instruct | — | — |
| SDPO | Qwen2.5-1.5B-Instruct | — | — |

🚧 **Experiments pending** — results will be filled in after training completes.

## Directory Structure

```
04_self_opd/
├── train_data/                  # RL prompt parquet (built by launch.sh)
├── methods/
│   └── sdpo/
│       ├── launch.sh            # end-to-end SDPO pipeline
│       ├── checkpoints/         # FSDP + merged HF checkpoints
│       └── results/             # eval json/details
└── README.md
```

## Notes

- SDPO uses the model's own high-reward trajectories as demonstrations for
  self-teaching; no external teacher model is required.
- The self-teacher is the *live* policy fed the reprompted context (paper's
  core formulation). EMA regularization is available via
  `self_distillation.teacher_regularization=ema` if a separate teacher module
  is provided, but is not required for the core method.
