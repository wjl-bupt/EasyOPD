# SDPO — Self-Distillation Policy Optimization

> Paper: [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
> (Hübotter et al., 2026) · Code: https://github.com/lasgroup/SDPO

SDPO augments on-policy RL (GRPO) with **self-distillation**: for each prompt the
policy samples a *group* of rollouts; a failed rollout is reprompted with a
correct demonstration from a successful rollout in the **same group** (and/or
environment feedback), and an **EMA copy of the policy** (the **self-teacher**
`π̃(·|x, f)`, initialised from the base model and EMA-updated towards the policy
each step) re-scores the failed rollout's original response under that
feedback-informed context. The student `π(·|x)` is distilled towards the
stop-gradient self-teacher (paper Eq. 1):

```
L_SDPO = Σ_t D_JSD^alpha( π(·|x, y_<t) ‖ stopgrad(π̃(·|x, f, y_<t)) )
```

`alpha` interpolates forward/reverse KL (0.5 = symmetric JSD, paper-recommended
for stability). **This is a faithful reimplementation of the lasgroup/SDPO
reference**: samples without a usable self-teacher contribute **zero gradient**
(no GRPO fallback), and the per-token distillation loss is multiplied by
token-level rollout-correction IS weights.

## Files

| File | Contents |
|------|----------|
| `core.py` | `compute_sdpo_self_distillation_loss` (full-logit / top-K generalized-JSD, IS clip, rollout-IS weights), `build_reprompt_text`, `select_demonstration`, `compute_ema_update`, `build_sdpo_teacher_inputs` (reprompt teacher batch), `compute_sdpo_actor_loss` (EMA-teacher forward + loss, no fallback), `compute_sdpo_loss` (hook wrapper), `remove_thinking_from_text`, `sequence_rewards` |
| `hooks.py` | `SDPOLossHook` (LossHook), `SDPOTeacherSidecarHook` (TeacherSidecarHook) |
| `__init__.py` | `@register_method("sdpo")` + `SDPOMethod` metadata |

## verl touchpoints (`# [EasyOPD:SDPO]`)

- `verl/workers/actor/dp_actor.py` — SDPO loss branch (`compute_sdpo_actor_loss`),
  `_forward_micro_batch(module=...)` for the teacher forward, `_update_teacher`
  (EMA), and `teacher_module` field.
- `verl/workers/fsdp_workers.py` — builds the EMA self-teacher module for SDPO
  (role="ref", FSDP CPUOffload), assigned to `self.actor.teacher_module`.
- `verl/trainer/ppo/ray_trainer.py` — `_maybe_build_sdpo_batch` builds the
  reprompt self-teacher batch after reward; rollout correction computes
  `rollout_is_weights` and adds them to the batch.
- `verl/trainer/ppo/rollout_corr_helper.py` — rollout-correction IS weights
  (training-vs-rollout policy mismatch), matching the reference.

## Implementation note: logit-level top-K distillation

This implementation uses the paper-default **logit-level top-K** generalized-JSD
(reference `full_logit_distillation=True`, `distillation_topk=100`): the
self-teacher is gathered at the **student's** top-K vocab indices (+ a tail
bucket) and the JSD is computed over that support. A token-level fallback (reverse
KL on chosen tokens) is used only when top-K extraction is unavailable (e.g.
under ulysses sequence parallelism).

## Usage

```python
from easyopd import EasyOPD
m = EasyOPD.from_hparams("sdpo")          # default config: easyopd/config/sdpo.yaml
```

Training (single 8-GPU node):

```bash
bash experiments/04_self_opd/methods/sdpo/launch.sh
# or the example launcher:
bash examples/sdpo/run_sdpo.sh
```

Key knobs (under `actor_rollout_ref.actor.self_distillation`): `alpha`,
`full_logit_distillation`, `distillation_topk`, `distillation_add_tail`,
`is_clip`, `success_reward_threshold`, `teacher_regularization` (`ema`),
`teacher_update_rate`, `dont_reprompt_on_self_success`,
`remove_thinking_from_demonstration`, `include_environment_feedback`,
`max_reprompt_len`. Rollout correction is configured under
`algorithm.rollout_correction` (`rollout_is=token`, `rollout_is_threshold=2.0`).
Requires `actor_rollout_ref.rollout.n > 1`.
