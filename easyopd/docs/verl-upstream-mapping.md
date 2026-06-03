# EasyOPD ↔ Upstream verl Distillation — Field Naming Map

This map exists so that, when EasyOPD eventually rebases onto an upstream
verl that includes `verl/trainer/distillation/`, users will not have to
edit their yaml configs. We keep canonical field names *aligned* with the
upstream `DistillationLossConfig` while accepting legacy EasyOPD names as
aliases.

## Field Mapping

| EasyOPD canonical | Upstream verl `DistillationLossConfig` | EasyOPD legacy alias |
|---|---|---|
| `loss_mode` | `loss_mode` | — |
| `topk` | `topk` | `kl_topk` |
| `use_task_rewards` | `use_task_rewards` | — |
| `distillation_loss_coef` | `distillation_loss_coef` | `kd_coef`, `kl_coef` |
| `loss_max_clamp` | `loss_max_clamp` | — |
| `log_prob_min_clamp` | `log_prob_min_clamp` | — |
| `use_policy_gradient` | `use_policy_gradient` | — |
| `policy_loss_mode` | `policy_loss_mode` | — |
| `clip_ratio` / `clip_ratio_low` / `clip_ratio_high` | same | — |

## Teacher-Manager Hook Signatures

| EasyOPD hook | Upstream `AsyncTeacherLLMServerManager` analog |
|---|---|
| `TeacherSidecarHook.teacher_forward(batch, teacher_model, config, **kwargs)` | `generate_with_teacher(prompt_batch)` (logits/log-probs branch) |
| _planned_ `TeacherSidecarHook.get_teacher_logprobs` | `get_teacher_logprobs(prompt_batch, response_batch)` |

> NOTE: We intentionally do **not** require Hook implementations to
> inherit from upstream classes. Upstream targets vllm 0.10+; we are on
> vllm 0.19 with vendored verl 0.5.0.dev. Names align, internals don't.

## Acceptance Strategy

`HookDispatcher.from_config` should accept either canonical or legacy
field names with a one-time `DeprecationWarning` per legacy field, and
log a hint that the new (upstream-compatible) name is preferred. The
warning text should include both names so users can mechanically migrate
yaml files.

## When We Rebase

The future verl rebase will:

1. Drop EasyOPD's redundant teacher-manager wiring in favor of upstream
   `AsyncTeacherLLMServerManager` for the 4 methods that fit (vopd,
   opcd, g_opd, sod).
2. Keep EasyOPD's Hook layer intact for the other 8 methods (opsa,
   ropd, gad, lightning_opd, gkd, sdpo, simple, simct).
3. Update `setup.py` / `requirements.txt` to reflect the rebased verl
   version. **This is a separate workstream from the Hook unification.**
