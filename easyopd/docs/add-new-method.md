# Adding a New OPD Method to EasyOPD

This guide shows you, in **4 steps**, how to plug a new OPD method into
EasyOPD without touching a single line of `verl/`.

The full reference example is `easyopd/methods/echo_kd/`.

## Prerequisites

You need three pieces:

- `easyopd/methods/<your_method>/__init__.py` — registers the method.
- `easyopd/methods/<your_method>/hooks.py` — adapts your algorithm into
  the 5 stable hook types.
- `easyopd/methods/<your_method>/config.yaml` — the user-facing config.

## The 4 Steps

### Step 1 — Implement your hook adapter(s)

Pick the appropriate hook type(s) and write a small adapter class:

| Hook | Use when… | Signature |
|---|---|---|
| `LossHook` | You compute a loss term added to the actor objective | `compute_loss(student_logits, teacher_logits, mask, config, **kwargs) -> (loss, metrics)` |
| `RolloutHook` | You attach metadata/process the batch after rollout | `on_rollout_end(batch, config, **kwargs) -> batch` |
| `RewardHook` | You score generations with a custom reward | `compute_reward(batch, config, **kwargs) -> rewards` |
| `AlignmentHook` | You build cross-tokenizer alignment data | `build_alignment(student_tokenizer, teacher_tokenizer, input_ids, config, **kwargs)` |
| `TeacherSidecarHook` | You define how the teacher forward pass runs | `teacher_forward(batch, teacher_model, config, **kwargs)` |

Example (`easyopd/methods/echo_kd/hooks.py`):

```python
import torch

class EchoKDLossHook:
    def compute_loss(self, student_logits, teacher_logits, mask, config=None, **kwargs):
        diff = (student_logits - teacher_logits) ** 2
        if diff.dim() == mask.dim() + 1:
            diff = diff.sum(dim=-1)
        loss = (diff * mask).sum() / mask.sum().clamp(min=1)
        return loss, {"echo_kd/mean_sq_diff": float(loss.detach().item())}
```

### Step 2 — Register the method

Use `@register_method` in your `__init__.py`:

```python
from easyopd.registry import register_method
from .hooks import EchoKDLossHook  # noqa: F401

@register_method("echo_kd")
class EchoKDMethod:
    name = "echo_kd"
    description = "Echo-KD: minimal MSE distillation demo."
```

If your method needs to accept legacy config aliases:

```python
@register_method("vision_opd", loss_mode_aliases=("vopd",))
class VisionOPDMethod: ...
```

### Step 3 — Write the user-facing yaml

```yaml
# easyopd/methods/echo_kd/config.yaml
easyopd:
  method:
    name: echo_kd
```

### Step 4 — Verify with a single test

```python
from easyopd.hook_dispatch import HookDispatcher

cfg = {"easyopd": {"method": {"name": "echo_kd"}}}
dispatcher = HookDispatcher.from_config(cfg)
assert dispatcher.enabled
assert dispatcher.hooks.has_loss
```

## What you do NOT need to change

- ❌ `verl/workers/actor/dp_actor.py`
- ❌ `verl/trainer/ppo/ray_trainer.py`
- ❌ `verl/workers/critic/dp_critic.py`
- ❌ `easyopd/hook_dispatch.py`
- ❌ `easyopd/registry.py`
- ❌ `easyopd/hooks.py`

## When You Genuinely Need a New Dispatch Boundary

If your method introduces a control-flow point that none of the 5 hooks
covers (e.g. you need a custom critic-side dispatch), the **only**
correct response is to add a new dispatch method to
`easyopd/hook_dispatch.py`, paired with a new Protocol in
`easyopd/hooks.py`. **Never** add an `if method_name == "..."` branch to
`verl/`.

## Troubleshooting

- **`MethodNotFoundError`**: ensure your `__init__.py` is imported by
  `easyopd/methods/__init__.py` (or relies on `auto_discover`).
- **`AttributeError: 'NoneType' object has no attribute 'compute_loss'`**:
  your hook class is named incorrectly. The auto-discovery looks for
  `<MethodPascalName>{Loss|Rollout|Reward|Alignment|TeacherSidecar}Hook`.
- **Dispatcher `enabled=False` even with config**: check `easyopd.method.name`
  is correctly nested in your yaml.
