# GAD Migration To EasyOPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate GAD (Generative Adversarial Distillation, arXiv:2511.10643) into EasyOPD as a first-class `easyopd.methods.gad` method with the smallest possible verl-side footprint, following the ROPD-style modular pattern that the `simct` / `simple` / `sod` methods already use.

**Architecture:** All GAD algorithm code lives under `easyopd/methods/gad/` as small focused modules (`core.py`, `critic_forward.py`, `critic_update.py`, `config.py`, `data_contract.py`). verl changes are bracketed by `# ============ [EasyOPD:GAD] ... # ============ [EasyOPD:GAD] End ============` comments, confined to **`verl/workers/critic/dp_critic.py` only** (3 wraps, ~19 lines). The dataset is required to provide four teacher-side tensor batch keys (see Data contract); they survive the trainer's existing pop/repeat/union flow without any `ray_trainer.py` modification. The critic forward is repurposed to emit a last-token-only score, which is consumed as a token-level reward by the standard PPO advantage path. The critic's `update_critic` is dispatched to GAD's Bradley-Terry pairwise loss when `gad.enable=true`.

**Tech Stack:** Python 3.10+, Bash, Hydra, pytest, PyTorch (CPU for tests), verl's existing critic worker / DataProto, EasyOPD method package convention (`METHOD = XxxMethod()` + `register()`).

---

## Spec ↔ Plan reconciliation

The spec (`docs/superpowers/specs/2026-05-27-gad-migration-to-easyopd-design.md`, commit `1fc73276`) describes 4 logical verl entry points (5 comment wraps, 10 lines) spread across `ray_trainer.py` and `dp_critic.py`. Closer reading of the **current** EasyOPD trainer revealed that ❶ (extend `batch_keys_to_pop`) and ❷ (rollout pass-through) are unnecessary:

- `_get_gen_batch` at `verl/trainer/ppo/ray_trainer.py:808-820` only pops three specific tensor keys (`input_ids`, `attention_mask`, `position_ids`). Any other tensor key in the dataset's batch dict (including the four GAD teacher-side keys listed below) **stays in `batch`** and survives the subsequent `batch.repeat(...)` + `batch.union(gen_batch_output)` flow at lines 1346-1347.
- Therefore no `ray_trainer.py` modification is needed. The plan drops the `rollout_passthrough.py` module entirely and reduces verl touches to **only `dp_critic.py`** (3 wraps in 2 methods, ~19 lines, still in one file).

This is a strict reduction of verl-side footprint, fully consistent with the spec's stated principle ("verl 主干尽量不动"). The non-goal "rollout module modification" remains honored a fortiori.

Other spec text remains accurate:
- `core.py` / `critic_forward.py` / `critic_update.py` / `config.py` / `data_contract.py` content is unchanged.
- The dispatch surface in `dp_critic.update_critic` and the `compute_teacher` kwarg in `_forward_micro_batch` are unchanged.
- The test list shrinks by one file (`test_rollout_passthrough.py` is dropped) and `test_no_drift.py`'s expected line count drops to 6 lines (3 wraps × 2).

**Second reconciliation — teacher key count.** The spec mentions "the dataset must carry `teacher_response`" (singular concept). Closer reading of the upstream `gad`-branch `_forward_micro_batch` (`/mnt/d/Area/DL/projects/research/YTianZHU-verl-gad/verl/workers/critic/dp_critic.py:58-75`) reveals that the critic forward needs **four** teacher-side tensor keys, not one:

- `teacher_input_ids` — the full critic input (prompt + teacher response), shape `(B, T_full_t)`
- `teacher_attention_mask` — attention mask for the full input, shape `(B, T_full_t)`
- `teacher_position_ids` — position ids for the full input, shape `(B, T_full_t)`
- `teacher_response` — the teacher response portion alone, shape `(B, T_resp_t)` — used to derive `response_length` for the slicing tail

The plan adopts the spec's design philosophy ("the dataset provides what the algorithm needs") but lists all four keys explicitly in `GAD_BATCH_KEYS`, the `remap_to_teacher` swap table, the data-contract docs, and the user-facing README. This matches the upstream gad-fork's dataset layout (LMSYS-Chat-GPT-5-Chat-Response is shipped with all four).

The implementer should record this reconciliation in the first commit's message body and proceed with the simpler 3-wrap plan.

---

## File Structure

### Files to create

- `easyopd/methods/gad/__init__.py`
- `easyopd/methods/gad/core.py`
- `easyopd/methods/gad/data_contract.py`
- `easyopd/methods/gad/config.py`
- `easyopd/methods/gad/critic_forward.py`
- `easyopd/methods/gad/critic_update.py`
- `easyopd/methods/gad/README.md`
- `easyopd/config/gad/base.yaml`
- `examples/gad_trainer/README.md`
- `examples/gad_trainer/train_gad.sh`
- `docs/algo/gad.md`
- `tests/easyopd/gad/__init__.py`
- `tests/easyopd/gad/conftest.py`
- `tests/easyopd/gad/test_imports.py`
- `tests/easyopd/gad/test_registration.py`
- `tests/easyopd/gad/test_core_numeric.py`
- `tests/easyopd/gad/test_data_contract.py`
- `tests/easyopd/gad/test_config.py`
- `tests/easyopd/gad/test_critic_forward.py`
- `tests/easyopd/gad/test_critic_update_contract.py`
- `tests/easyopd/gad/test_no_drift.py`
- `tests/easyopd/gad/test_no_actor_changes.py`
- `tests/easyopd/gad/test_config_smoke.py`
- `tests/easyopd/gad/test_entrypoints.py`

### Files to modify

- `verl/workers/critic/dp_critic.py` (3 `[EasyOPD:GAD]` wraps in 2 methods)
- `docs/index.rst` (add `algo/gad.md` to the toctree)

### Files explicitly NOT modified

- `verl/trainer/ppo/ray_trainer.py` — kept clean (see spec↔plan reconciliation above)
- `verl/workers/actor/dp_actor.py` — actor unchanged; advantages come from existing flow
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` — no rollout changes
- `verl/trainer/ppo/core_algos.py` — BT loss lives in `easyopd.methods.gad.core`, not here
- Other `easyopd/methods/` packages — no cross-method dependency
- `pyproject.toml` / `setup.py` — no new package data needed (pure Python, no `.txt` resources)

### Source files to read during implementation (no edits)

- `docs/superpowers/specs/2026-05-27-gad-migration-to-easyopd-design.md` (the spec)
- `easyopd/methods/simct/__init__.py` (registration pattern reference)
- `easyopd/methods/sod/__init__.py` (registration pattern reference)
- `easyopd/methods/simple/__init__.py` (registration pattern reference)
- `verl/workers/critic/dp_critic.py` (target file for the 3 wraps; current state)
- `verl/trainer/ppo/ray_trainer.py:808-820, 1306-1350` (confirm teacher_response survives)
- `/mnt/d/Area/DL/projects/research/YTianZHU-verl-gad/verl/workers/critic/dp_critic.py:260-310` (source GAD update_critic body)
- `/mnt/d/Area/DL/projects/research/YTianZHU-verl-gad/verl/trainer/ppo/core_algos.py:832-836` (source BT loss)

---

## Task 1: Inventory and reconciliation note

**Files:**
- Read: `docs/superpowers/specs/2026-05-27-gad-migration-to-easyopd-design.md`
- Read: `easyopd/methods/simct/__init__.py`
- Read: `easyopd/methods/sod/__init__.py`
- Read: `verl/trainer/ppo/ray_trainer.py` lines 808-820 and 1300-1360
- Read: `verl/workers/critic/dp_critic.py` lines 1-220

- [ ] **Step 1: Confirm `_get_gen_batch` only pops three tensor keys**

```bash
sed -n '808,820p' verl/trainer/ppo/ray_trainer.py
```
Expected: `batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]` and no other tensor keys.

- [ ] **Step 2: Confirm `batch.repeat(...).union(gen_batch_output)` is the post-rollout merge**

```bash
sed -n '1344,1350p' verl/trainer/ppo/ray_trainer.py
```
Expected: lines containing `batch = batch.repeat(...)` followed by `batch = batch.union(gen_batch_output)`.

- [ ] **Step 3: Record the file list locked by this plan**

There is no scratch file to commit yet. Confirm in your head:
```text
Create:  easyopd/methods/gad/{__init__,core,data_contract,config,critic_forward,critic_update,README}.{py,md}
Create:  easyopd/config/gad/base.yaml
Create:  examples/gad_trainer/{README.md,train_gad.sh}
Create:  docs/algo/gad.md
Create:  tests/easyopd/gad/{__init__,conftest,test_*.py}        (12 test files)
Modify:  verl/workers/critic/dp_critic.py                       (3 [EasyOPD:GAD] wraps)
Modify:  docs/index.rst                                         (one toctree line)
```

- [ ] **Step 4: Commit a placeholder for the plan file itself (this document)**

```bash
git add docs/superpowers/plans/2026-05-27-gad-migration-to-easyopd-plan.md
git commit -m "docs: add GAD migration implementation plan

Plan diverges from spec §verl接入点 by dropping ❶ and ❷:
_get_gen_batch only pops 3 fixed tensor keys, so teacher_response
naturally survives in batch through the rollout flow. verl footprint
reduces from 4 logical entries (5 wraps) to 3 wraps in 1 file
(dp_critic.py only).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Scaffold the `easyopd/methods/gad/` package skeleton

**Files:**
- Create: `easyopd/methods/gad/__init__.py`
- Create: `easyopd/methods/gad/README.md`
- Create: `tests/easyopd/gad/__init__.py`
- Create: `tests/easyopd/gad/test_imports.py`
- Create: `tests/easyopd/gad/test_registration.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/easyopd/gad/test_imports.py`:

```python
"""Importability smoke tests for easyopd.methods.gad."""


def test_package_imports():
    import easyopd.methods.gad  # noqa: F401


def test_method_metadata_present():
    from easyopd.methods.gad import METHOD

    assert METHOD.name == "gad"
    assert "verl/workers/critic/dp_critic.py" in METHOD.verl_modified_files
    assert METHOD.description  # non-empty
```

Create empty `tests/easyopd/gad/__init__.py`.

- [ ] **Step 2: Run the test and confirm it fails on missing import**

Run: `pytest tests/easyopd/gad/test_imports.py -q`
Expected: `ModuleNotFoundError: No module named 'easyopd.methods.gad'`

- [ ] **Step 3: Write the failing registration test**

Create `tests/easyopd/gad/test_registration.py`:

```python
"""Registration / metadata structure tests for easyopd.methods.gad."""

from dataclasses import is_dataclass


def test_method_dataclass_is_frozen():
    from easyopd.methods.gad import METHOD, GADMethod

    assert is_dataclass(GADMethod)
    assert isinstance(METHOD, GADMethod)


def test_register_is_callable_and_idempotent():
    from easyopd.methods.gad import register

    register()
    register()  # must not raise on second call
```

- [ ] **Step 4: Run the registration test and confirm it fails**

Run: `pytest tests/easyopd/gad/test_registration.py -q`
Expected: same `ModuleNotFoundError`.

- [ ] **Step 5: Create the package `__init__.py` with metadata**

Create `easyopd/methods/gad/__init__.py`:

```python
# Copyright 2026 EasyOPD Contributors
#
# gad: Generative Adversarial Distillation (arXiv:2511.10643).
# Hacks verl's critic into a Bradley-Terry discriminator that scores
# student vs teacher responses, with the standard PPO actor consuming
# the discriminator output as a token-level reward signal.
#
# Verl files this method touches (all changes are wrapped in
# `# ============ [EasyOPD:GAD] ... # ============ [EasyOPD:GAD] End ============`
# comment markers):
#   * verl/workers/critic/dp_critic.py
#       - _forward_micro_batch: accept `compute_teacher` kwarg, swap
#         input keys via remap_to_teacher, and reduce to last-token-only.
#       - update_critic: dispatch to easyopd.methods.gad.critic_update
#         when gad.enable=true.

from dataclasses import dataclass


@dataclass(frozen=True)
class GADMethod:
    """Static metadata describing the EasyOPD `gad` method."""

    name: str = "gad"
    verl_modified_files: tuple = ("verl/workers/critic/dp_critic.py",)
    paper_url: str = "https://arxiv.org/abs/2511.10643"
    description: str = (
        "Generative Adversarial Distillation: repurposes the PPO critic "
        "as a Bradley-Terry discriminator over student vs teacher "
        "responses; the discriminator's last-token output drives the "
        "standard PPO advantage / actor update."
    )


METHOD = GADMethod()


def register() -> None:
    """Idempotent registration entry point.

    GAD has no global registry side-effect to perform (unlike the loss-
    mode registration done by `simple`/`simct`): dispatch is config-
    driven via `is_gad_enabled(cfg)` checks in `verl/workers/critic/
    dp_critic.py`. This function exists so the method follows the same
    public shape as the other EasyOPD methods, and may be wired into a
    future central registry without changing call sites.
    """
    return None


__all__ = ["METHOD", "GADMethod", "register"]
```

- [ ] **Step 6: Create a minimal README placeholder**

Create `easyopd/methods/gad/README.md`:

```markdown
# GAD: Generative Adversarial Distillation

## Method summary

GAD repurposes the verl PPO critic as a Bradley-Terry discriminator over
student vs teacher responses. The discriminator's last-token score is
consumed by the standard PPO advantage / actor path as a token-level
reward. See `docs/algo/gad.md` for the full method description, and
`docs/superpowers/specs/2026-05-27-gad-migration-to-easyopd-design.md`
for the integration design.

Paper: https://arxiv.org/abs/2511.10643

## verl files modified

| File | What changes | Why |
|------|--------------|-----|
| `verl/workers/critic/dp_critic.py` | `_forward_micro_batch` accepts `compute_teacher` kwarg, swaps input keys to the four teacher_* keys when set, reduces output to last-token-only | Critic becomes a discriminator; the only relevant scalar is the seq-level score at the last token |
| `verl/workers/critic/dp_critic.py` | `update_critic` dispatches to `easyopd.methods.gad.critic_update.update_critic_step` when `gad.enable=true` | Replaces MSE value loss with Bradley-Terry pairwise loss |

All edits are wrapped in `# ============ [EasyOPD:GAD] ... # ============ [EasyOPD:GAD] End ============` comments.

## Data contract

Training samples must provide four extra tensor batch keys:

- `teacher_input_ids`: full critic input for the teacher (prompt + teacher response), shape `[B, T_full_t]`
- `teacher_attention_mask`: attention mask matching `teacher_input_ids`, shape `[B, T_full_t]`
- `teacher_position_ids`: position ids matching `teacher_input_ids`, shape `[B, T_full_t]`
- `teacher_response`: just the teacher response tokens (used to derive `response_length`), shape `[B, T_resp_t]`

All four shapes are independent of the student's `T_s`. The trainer's
existing `_get_gen_batch` only pops `input_ids / attention_mask /
position_ids`, so these extra tensor keys survive automatically into
the `batch` consumed by `compute_values` and `update_critic`.

## Reproduction outline

1. Prepare a parquet dataset whose rows include the four teacher-side
   tensor fields (`teacher_input_ids`, `teacher_attention_mask`,
   `teacher_position_ids`, `teacher_response`). The upstream paper used GPT-5-Chat responses on LMSYS prompts
   (see `https://huggingface.co/datasets/ytz20/LMSYS-Chat-GPT-5-Chat-Response`
   and `microsoft/LMOps/gad/tools/export_lmsys_parquet.py` for an example).
2. Provide a pretrained discriminator checkpoint and pass its path via
   `gad.discriminator_init_path=<path>`. Use a discriminator that has
   already been warmed up (e.g. SFT on teacher responses).
3. Launch with `bash examples/gad_trainer/train_gad.sh ...`.
```

- [ ] **Step 7: Run the import + registration tests and confirm they pass**

Run: `pytest tests/easyopd/gad/test_imports.py tests/easyopd/gad/test_registration.py -q`
Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add easyopd/methods/gad/__init__.py easyopd/methods/gad/README.md \
        tests/easyopd/gad/__init__.py \
        tests/easyopd/gad/test_imports.py tests/easyopd/gad/test_registration.py
git commit -m "feat(gad): scaffold easyopd.methods.gad package with METHOD metadata

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Implement `core.py` (pure tensor functions)

**Files:**
- Create: `easyopd/methods/gad/core.py`
- Create: `tests/easyopd/gad/test_core_numeric.py`

- [ ] **Step 1: Write the failing core numeric tests**

Create `tests/easyopd/gad/test_core_numeric.py`:

```python
"""CPU numeric tests for easyopd.methods.gad.core pure functions."""

import math

import torch


def test_summed_reward_applies_mask():
    from easyopd.methods.gad.core import summed_reward

    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    out = summed_reward(values, mask)

    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([3.0, 11.0]))


def test_compute_discriminator_loss_matches_closed_form():
    from easyopd.methods.gad.core import compute_discriminator_loss

    student = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    teacher = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    smask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    tmask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    loss = compute_discriminator_loss(student, teacher, smask, tmask)

    # teacher_sum - student_sum = [2, 2]; -mean(log_sigmoid([2,2]))
    expected = -math.log(1.0 / (1.0 + math.exp(-2.0)))
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_discriminator_accuracy_perfect_and_inverted():
    from easyopd.methods.gad.core import discriminator_accuracy

    student = torch.tensor([[1.0], [1.0]])
    teacher = torch.tensor([[2.0], [2.0]])
    mask = torch.tensor([[1.0], [1.0]])

    acc_good = discriminator_accuracy(student, teacher, mask, mask)
    assert acc_good == 1.0

    acc_bad = discriminator_accuracy(teacher, student, mask, mask)
    assert acc_bad == 0.0


def test_last_token_only_keeps_only_final_valid_position():
    from easyopd.methods.gad.core import last_token_only

    values = torch.tensor([[0.5, 1.5, 2.5, 3.5], [10.0, 20.0, 30.0, 40.0]])
    # Row 0 has 3 valid tokens (last valid at index 2); row 1 has 4 (last at 3).
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    out = last_token_only(values, response_mask)

    expected = torch.tensor([[0.0, 0.0, 2.5, 0.0], [0.0, 0.0, 0.0, 40.0]])
    assert torch.allclose(out, expected)


def test_last_token_only_handles_all_zero_mask_row():
    from easyopd.methods.gad.core import last_token_only

    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[0.0, 0.0, 0.0]])
    out = last_token_only(values, mask)

    # When no valid token, output must be all zeros (no spurious score).
    assert torch.allclose(out, torch.zeros_like(values))
```

- [ ] **Step 2: Run and confirm tests fail on missing module**

Run: `pytest tests/easyopd/gad/test_core_numeric.py -q`
Expected: `ModuleNotFoundError: No module named 'easyopd.methods.gad.core'`

- [ ] **Step 3: Implement `core.py`**

Create `easyopd/methods/gad/core.py`:

```python
"""Pure tensor primitives for GAD.

No verl, no DataProto, no I/O. Everything in this module is unit-
testable on CPU with small tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def summed_reward(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sum masked per-token scores into a per-sequence scalar.

    Args:
        values: shape (B, T), per-token discriminator output.
        response_mask: shape (B, T), 1.0 on valid positions, 0.0 elsewhere.

    Returns:
        Tensor of shape (B,).
    """
    return (values * response_mask).sum(dim=-1)


def compute_discriminator_loss(
    student_vpreds: torch.Tensor,
    teacher_vpreds: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise loss for discriminator.

    Loss = -mean(log_sigmoid(teacher_reward - student_reward))
    where each reward is the masked sum over its sequence.

    Lifted from `YTianZHU/verl@gad:verl/trainer/ppo/core_algos.py:832-836`.
    """
    teacher_reward = summed_reward(teacher_vpreds, teacher_response_mask)
    student_reward = summed_reward(student_vpreds, response_mask)
    return -F.logsigmoid(teacher_reward - student_reward).mean()


def discriminator_accuracy(
    student_vpreds: torch.Tensor,
    teacher_vpreds: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
) -> float:
    """Fraction of (s, t) pairs where teacher_sum > student_sum."""
    teacher_reward = summed_reward(teacher_vpreds, teacher_response_mask)
    student_reward = summed_reward(student_vpreds, response_mask)
    return (teacher_reward > student_reward).float().mean().item()


def last_token_only(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Zero out every position except the last valid one in each row.

    Used to turn the critic's per-token output into a sequence-level
    score localized at the final response token. Rows with an all-zero
    mask become all-zero rows (no spurious score).
    """
    response_lengths = response_mask.sum(dim=1).long()  # (B,)
    has_valid = response_lengths > 0  # (B,)
    last_token_indices = (response_lengths - 1).clamp(min=0)  # (B,)

    last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
    last_token_mask[batch_indices, last_token_indices] = True
    last_token_mask[~has_valid] = False  # rows with no valid token: keep all zeros

    return values * last_token_mask.type_as(values)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/easyopd/gad/test_core_numeric.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add easyopd/methods/gad/core.py tests/easyopd/gad/test_core_numeric.py
git commit -m "feat(gad): add core tensor primitives (BT loss, last_token_only)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Implement `data_contract.py`

**Files:**
- Create: `easyopd/methods/gad/data_contract.py`
- Create: `tests/easyopd/gad/test_data_contract.py`

- [ ] **Step 1: Write the failing data-contract tests**

Create `tests/easyopd/gad/test_data_contract.py`:

```python
"""Tests for the GAD batch data contract."""

import pytest
import torch


def _make_batch_dict(with_teacher: bool = True, mismatch_batch_dim: bool = False):
    bsz = 4
    out = {
        "input_ids": torch.zeros(bsz, 8, dtype=torch.long),
        "attention_mask": torch.ones(bsz, 8, dtype=torch.long),
        "position_ids": torch.zeros(bsz, 8, dtype=torch.long),
        "responses": torch.zeros(bsz, 4, dtype=torch.long),
    }
    if with_teacher:
        tbsz = bsz + (1 if mismatch_batch_dim else 0)
        out["teacher_input_ids"] = torch.zeros(tbsz, 9, dtype=torch.long)
        out["teacher_attention_mask"] = torch.ones(tbsz, 9, dtype=torch.long)
        out["teacher_position_ids"] = torch.zeros(tbsz, 9, dtype=torch.long)
        out["teacher_response"] = torch.zeros(tbsz, 6, dtype=torch.long)
    return out


def test_keys_constant_matches_spec():
    from easyopd.methods.gad.data_contract import GAD_BATCH_KEYS

    assert GAD_BATCH_KEYS == (
        "teacher_input_ids",
        "teacher_attention_mask",
        "teacher_position_ids",
        "teacher_response",
    )


def test_validate_accepts_well_formed_batch():
    from easyopd.methods.gad.data_contract import validate_gad_batch

    validate_gad_batch(_make_batch_dict(with_teacher=True))


def test_validate_rejects_missing_keys():
    from easyopd.methods.gad.data_contract import (
        GADBatchContractError,
        validate_gad_batch,
    )

    bad = _make_batch_dict(with_teacher=True)
    del bad["teacher_input_ids"]
    del bad["teacher_position_ids"]
    with pytest.raises(GADBatchContractError) as ei:
        validate_gad_batch(bad)
    msg = str(ei.value)
    assert "teacher_input_ids" in msg
    assert "teacher_position_ids" in msg
    assert "docs/algo/gad.md" in msg


def test_validate_rejects_mismatched_batch_dim():
    from easyopd.methods.gad.data_contract import (
        GADBatchContractError,
        validate_gad_batch,
    )

    with pytest.raises(GADBatchContractError) as ei:
        validate_gad_batch(_make_batch_dict(with_teacher=True, mismatch_batch_dim=True))
    assert "batch dim" in str(ei.value).lower()
```

- [ ] **Step 2: Run and confirm tests fail on missing module**

Run: `pytest tests/easyopd/gad/test_data_contract.py -q`
Expected: 4 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `data_contract.py`**

Create `easyopd/methods/gad/data_contract.py`:

```python
"""Data contract validation for GAD batches.

GAD requires the dataset to provide FOUR teacher-side tensor batch keys
(see GAD_BATCH_KEYS below). These keys are NOT popped into the
gen_batch by verl's `_get_gen_batch` (which only pops input_ids /
attention_mask / position_ids), so they survive into the post-rollout
`batch` consumed by `compute_values` and `update_critic`.

This module validates the contract once per training run (called at
the start of GAD's `update_critic_step`).
"""

from __future__ import annotations

from typing import Mapping

GAD_BATCH_KEYS: tuple[str, ...] = (
    "teacher_input_ids",
    "teacher_attention_mask",
    "teacher_position_ids",
    "teacher_response",
)


class GADBatchContractError(ValueError):
    """Raised when GAD is enabled but the batch violates the data contract."""


def validate_gad_batch(batch_like: Mapping) -> None:
    """Verify a batch dict (or batch.batch from a DataProto) satisfies the GAD contract.

    Checks performed:
      - All four keys in `GAD_BATCH_KEYS` are present.
      - Their batch dimensions match `input_ids`' batch dimension when
        `input_ids` is present in the same dict.

    Args:
        batch_like: anything supporting `__contains__`, `__getitem__`, and `keys()`
            over tensor-shaped values. In practice either a plain dict or a
            DataProto's `.batch` TensorDict.

    Raises:
        GADBatchContractError with a message that names the missing or
        mis-shaped keys and points to docs/algo/gad.md for the schema.
    """
    missing = [k for k in GAD_BATCH_KEYS if k not in batch_like]
    if missing:
        present = sorted(batch_like.keys())
        raise GADBatchContractError(
            f"GAD is enabled but batch is missing required keys. "
            f"Missing: {missing}. Batch has keys: {present}. "
            "See docs/algo/gad.md §Data preparation for the required schema."
        )

    if "input_ids" in batch_like:
        ref_bsz = batch_like["input_ids"].shape[0]
        for k in GAD_BATCH_KEYS:
            actual = batch_like[k].shape[0]
            if actual != ref_bsz:
                raise GADBatchContractError(
                    f"GAD batch dim mismatch for '{k}': "
                    f"expected {ref_bsz} (input_ids batch dim), got {actual}."
                )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/easyopd/gad/test_data_contract.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add easyopd/methods/gad/data_contract.py tests/easyopd/gad/test_data_contract.py
git commit -m "feat(gad): add data contract validation for teacher_response keys

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement `config.py` (GADConfig + is_gad_enabled)

**Files:**
- Create: `easyopd/methods/gad/config.py`
- Create: `tests/easyopd/gad/test_config.py`

- [ ] **Step 1: Write the failing config tests**

Create `tests/easyopd/gad/test_config.py`:

```python
"""Tests for GADConfig loading and validation."""

import pytest
from omegaconf import OmegaConf


def _cfg(**gad_overrides):
    base = {
        "gad": {
            "enable": False,
            "discriminator_init_path": None,
        },
        "reward_model": {"enable": False},
    }
    base["gad"].update(gad_overrides)
    return OmegaConf.create(base)


def test_is_gad_enabled_false_by_default():
    from easyopd.methods.gad.config import is_gad_enabled

    cfg = OmegaConf.create({"trainer": {"foo": 1}})
    assert is_gad_enabled(cfg) is False


def test_is_gad_enabled_true_when_flag_set():
    from easyopd.methods.gad.config import is_gad_enabled

    cfg = _cfg(enable=True, discriminator_init_path="/tmp/disc")
    assert is_gad_enabled(cfg) is True


def test_load_returns_dataclass_when_disabled():
    from easyopd.methods.gad.config import GADConfig

    cfg = _cfg(enable=False)
    gad_cfg = GADConfig.load_from_omegaconf(cfg)
    assert gad_cfg.enable is False
    assert gad_cfg.discriminator_init_path is None


def test_load_raises_when_enabled_without_path():
    from easyopd.methods.gad.config import GADConfig, GADConfigError

    cfg = _cfg(enable=True, discriminator_init_path=None)
    with pytest.raises(GADConfigError) as ei:
        GADConfig.load_from_omegaconf(cfg)
    assert "discriminator_init_path" in str(ei.value)


def test_load_collects_all_violations():
    from easyopd.methods.gad.config import GADConfig, GADConfigError

    cfg = OmegaConf.create(
        {
            "gad": {"enable": True, "discriminator_init_path": None},
            "reward_model": {"enable": True},
        }
    )
    with pytest.raises(GADConfigError) as ei:
        GADConfig.load_from_omegaconf(cfg)
    msg = str(ei.value)
    # Both problems must be reported, not just the first one.
    assert "discriminator_init_path" in msg
    assert "reward_model" in msg
    assert msg.count("\n") >= 1  # multiline message


def test_load_succeeds_when_enabled_with_path():
    from easyopd.methods.gad.config import GADConfig

    cfg = _cfg(enable=True, discriminator_init_path="/tmp/disc.ckpt")
    gad_cfg = GADConfig.load_from_omegaconf(cfg)
    assert gad_cfg.enable is True
    assert gad_cfg.discriminator_init_path == "/tmp/disc.ckpt"
```

- [ ] **Step 2: Run and confirm tests fail**

Run: `pytest tests/easyopd/gad/test_config.py -q`
Expected: 6 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `config.py`**

Create `easyopd/methods/gad/config.py`:

```python
"""GAD configuration dataclass and the canonical `is_gad_enabled` check.

`is_gad_enabled(cfg)` is the SINGLE entry point used by every
`[EasyOPD:GAD]` if-branch in verl. Grepping for that string locates all
integration points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from omegaconf import DictConfig, OmegaConf


class GADConfigError(ValueError):
    """Raised when GAD is enabled but the surrounding config is invalid."""


def is_gad_enabled(cfg: Any) -> bool:
    """Return True iff cfg.gad.enable is truthy.

    Defensive against missing `gad` node so verl runs without GAD config
    behave identically to before.
    """
    if cfg is None:
        return False
    gad_node = OmegaConf.select(cfg, "gad", default=None) if isinstance(cfg, DictConfig) else getattr(cfg, "gad", None)
    if gad_node is None:
        return False
    enable = OmegaConf.select(gad_node, "enable", default=False) if isinstance(gad_node, DictConfig) else getattr(gad_node, "enable", False)
    return bool(enable)


@dataclass(frozen=True)
class GADConfig:
    enable: bool = False
    discriminator_init_path: str | None = None
    metrics_prefix: str = "gad"

    @classmethod
    def load_from_omegaconf(cls, cfg: Any) -> "GADConfig":
        """Build and validate a GADConfig from the trainer's top-level cfg.

        Validation collects ALL violations and raises a single GADConfigError
        with a multi-line message, so the user can fix everything in one pass.
        """
        gad_node = OmegaConf.select(cfg, "gad", default=None) if isinstance(cfg, DictConfig) else getattr(cfg, "gad", None)
        enable = bool(OmegaConf.select(gad_node, "enable", default=False)) if gad_node is not None else False
        path = OmegaConf.select(gad_node, "discriminator_init_path", default=None) if gad_node is not None else None
        prefix = OmegaConf.select(gad_node, "metrics_prefix", default="gad") if gad_node is not None else "gad"

        if not enable:
            return cls(enable=False, discriminator_init_path=None, metrics_prefix=prefix or "gad")

        problems: List[str] = []

        if path in (None, "", "???"):
            problems.append(
                "gad.discriminator_init_path is required when gad.enable=true "
                f"(got {path!r})"
            )

        rm_enable = OmegaConf.select(cfg, "reward_model.enable", default=False)
        if bool(rm_enable):
            problems.append(
                "gad.enable=true is incompatible with reward_model.enable=true "
                "(GAD uses the critic as the reward source). Set reward_model.enable=false."
            )

        if problems:
            joined = "\n  - " + "\n  - ".join(problems)
            raise GADConfigError(f"GAD config has {len(problems)} problems:{joined}")

        return cls(enable=True, discriminator_init_path=str(path), metrics_prefix=prefix or "gad")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/easyopd/gad/test_config.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add easyopd/methods/gad/config.py tests/easyopd/gad/test_config.py
git commit -m "feat(gad): add GADConfig + is_gad_enabled with multi-error aggregation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement `critic_forward.py` (remap_to_teacher helper)

**Files:**
- Create: `easyopd/methods/gad/critic_forward.py`
- Create: `tests/easyopd/gad/conftest.py` (fixtures shared with later tasks)
- Create: `tests/easyopd/gad/test_critic_forward.py`

- [ ] **Step 1: Create shared test fixtures**

Create `tests/easyopd/gad/conftest.py`:

```python
"""Shared fixtures for GAD CPU contract tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch


@pytest.fixture
def tiny_micro_batch():
    """A dict-of-tensors mimicking a single micro-batch on CPU."""
    bsz, t_s, t_t = 2, 4, 5
    return {
        "input_ids": torch.arange(bsz * t_s).reshape(bsz, t_s).long(),
        "attention_mask": torch.ones(bsz, t_s, dtype=torch.long),
        "position_ids": torch.arange(t_s).unsqueeze(0).expand(bsz, t_s).long(),
        "responses": torch.arange(bsz * 2).reshape(bsz, 2).long(),  # response_length = 2
        "teacher_input_ids": (torch.arange(bsz * t_t) + 100).reshape(bsz, t_t).long(),
        "teacher_attention_mask": torch.ones(bsz, t_t, dtype=torch.long),
        "teacher_position_ids": torch.arange(t_t).unsqueeze(0).expand(bsz, t_t).long(),
        "teacher_response": (torch.arange(bsz * 3) + 200).reshape(bsz, 3).long(),  # teacher_response_length = 3
    }


@pytest.fixture
def constant_logits_module():
    """A torch.nn.Module returning a fixed-shape logits/values tensor.

    Mimics the minimal interface that dp_critic._forward_micro_batch
    uses: a callable that returns either a value head output or a model
    with `.logits`.
    """

    class _Out:
        def __init__(self, logits: torch.Tensor):
            self.logits = logits

    class _Mod(torch.nn.Module):
        def __init__(self, response_length: int):
            super().__init__()
            self.response_length = response_length
            self.calls: list[dict[str, Any]] = []

        def forward(self, **kwargs):
            self.calls.append({k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)})
            ids = kwargs["input_ids"]
            bsz, seqlen = ids.shape[0], ids.shape[-1]
            # constant per-token scalar (use 1.0 for student inputs, 2.0 if input shifted by 100, marker for teacher)
            base = 2.0 if ids.float().mean() >= 50 else 1.0
            return _Out(logits=torch.full((bsz, seqlen, 1), base))

    return _Mod
```

- [ ] **Step 2: Write the failing tests**

Create `tests/easyopd/gad/test_critic_forward.py`:

```python
"""Tests for easyopd.methods.gad.critic_forward."""

import torch


def test_remap_to_teacher_swaps_keys(tiny_micro_batch):
    from easyopd.methods.gad.critic_forward import remap_to_teacher

    remapped = remap_to_teacher(tiny_micro_batch)

    # Inputs come from the teacher_* keys.
    assert torch.equal(remapped["input_ids"], tiny_micro_batch["teacher_input_ids"])
    assert torch.equal(remapped["attention_mask"], tiny_micro_batch["teacher_attention_mask"])
    assert torch.equal(remapped["position_ids"], tiny_micro_batch["teacher_position_ids"])
    # `responses` is swapped too so verl's internal `response_length = responses.size(-1)`
    # picks up the teacher response length, not the student's.
    assert torch.equal(remapped["responses"], tiny_micro_batch["teacher_response"])


def test_remap_to_teacher_does_not_mutate_input(tiny_micro_batch):
    from easyopd.methods.gad.critic_forward import remap_to_teacher

    snapshot_ids = tiny_micro_batch["input_ids"].clone()
    snapshot_mask = tiny_micro_batch["attention_mask"].clone()
    snapshot_responses = tiny_micro_batch["responses"].clone()
    _ = remap_to_teacher(tiny_micro_batch)

    assert torch.equal(tiny_micro_batch["input_ids"], snapshot_ids)
    assert torch.equal(tiny_micro_batch["attention_mask"], snapshot_mask)
    assert torch.equal(tiny_micro_batch["responses"], snapshot_responses)


def test_remap_preserves_extra_keys(tiny_micro_batch):
    from easyopd.methods.gad.critic_forward import remap_to_teacher

    tiny_micro_batch["multi_modal_inputs"] = [{"foo": torch.zeros(1)}]
    remapped = remap_to_teacher(tiny_micro_batch)

    assert "multi_modal_inputs" in remapped
```

- [ ] **Step 3: Run and confirm tests fail**

Run: `pytest tests/easyopd/gad/test_critic_forward.py -q`
Expected: 3 failures with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `critic_forward.py`**

Create `easyopd/methods/gad/critic_forward.py`:

```python
"""Critic-forward adapter for GAD.

The verl critic's `_forward_micro_batch` reads `input_ids`,
`attention_mask`, `position_ids`, and `responses` from the micro-batch
dict (the last one is used to derive `response_length` for the slicing
tail). When GAD needs the critic to score the *teacher* response, we
swap all four keys to point at their `teacher_*` counterparts.

We never mutate the caller's dict.
"""

from __future__ import annotations

from typing import Mapping


_STUDENT_TO_TEACHER: dict[str, str] = {
    "input_ids": "teacher_input_ids",
    "attention_mask": "teacher_attention_mask",
    "position_ids": "teacher_position_ids",
    "responses": "teacher_response",
}


def remap_to_teacher(micro_batch: Mapping) -> dict:
    """Return a new dict where the student forward keys are sourced from teacher_* keys.

    Args:
        micro_batch: dict containing student keys (`input_ids`,
            `attention_mask`, `position_ids`, `responses`) AND teacher
            keys (`teacher_input_ids`, `teacher_attention_mask`,
            `teacher_position_ids`, `teacher_response`). May contain
            other keys (e.g. `multi_modal_inputs`) which are passed
            through unchanged.

    Returns:
        A new dict suitable for passing to the verl critic's existing
        forward path. The critic will read `responses` to compute
        `response_length`, which will now reflect the teacher response.
    """
    remapped = dict(micro_batch)  # shallow copy
    for student_key, teacher_key in _STUDENT_TO_TEACHER.items():
        remapped[student_key] = micro_batch[teacher_key]
    return remapped
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `pytest tests/easyopd/gad/test_critic_forward.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add easyopd/methods/gad/critic_forward.py \
        tests/easyopd/gad/conftest.py \
        tests/easyopd/gad/test_critic_forward.py
git commit -m "feat(gad): add critic forward adapter (remap_to_teacher)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Implement `critic_update.py` (update_critic_step)

**Files:**
- Create: `easyopd/methods/gad/critic_update.py`
- Create: `tests/easyopd/gad/test_critic_update_contract.py`

- [ ] **Step 1: Write the failing contract tests**

Create `tests/easyopd/gad/test_critic_update_contract.py`:

```python
"""Contract tests for easyopd.methods.gad.critic_update.update_critic_step.

We do not test numerical equivalence to the verl PPO critic update.
We test that the function:
  (a) runs student and teacher forwards via the injected critic worker,
  (b) calls loss.backward and an optimizer step,
  (c) returns a DataProto-like result with metrics containing the expected keys.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf


class _FakeDataProto:
    """A minimal stand-in for verl.protocol.DataProto in CPU tests."""

    def __init__(self, batch: dict, meta_info: dict | None = None):
        self.batch = dict(batch)
        self.non_tensor_batch: dict = {}
        self.meta_info = meta_info or {}

    def select(self, batch_keys=None, non_tensor_batch_keys=None):
        if batch_keys is None:
            return _FakeDataProto(self.batch, self.meta_info)
        return _FakeDataProto({k: self.batch[k] for k in batch_keys if k in self.batch}, self.meta_info)

    def split(self, micro_batch_size: int):
        # Single split for the contract test.
        return [self]

    def to(self, device):
        return self


def _make_worker(monkeypatch):
    """Build a mock critic worker the way dp_critic uses self."""

    optimizer = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()

    # Make _forward_micro_batch return a tensor where the LAST token has 1.0
    # (student) or 2.0 (teacher), as if last_token_only has already been applied.
    def fake_forward(self, micro_batch, *, compute_teacher: bool = False):
        bsz, t = micro_batch["input_ids"].shape[0], micro_batch["input_ids"].shape[-1]
        out = torch.zeros(bsz, t, requires_grad=True)
        with torch.no_grad():
            scale = 2.0 if compute_teacher else 1.0
            out_l = out.clone()
            out_l[:, -1] = scale
        out_l.requires_grad_(True)
        return out_l

    worker = SimpleNamespace(
        config=OmegaConf.create(
            {
                "gad": {"enable": True, "discriminator_init_path": "/tmp/x"},
                "ppo_mini_batch_size": 2,
                "use_dynamic_bsz": False,
                "ppo_micro_batch_size_per_gpu": 2,
            }
        ),
        critic_optimizer=optimizer,
        critic_module=MagicMock(),  # GradientAccumulator-like; train/eval will be called
        gradient_accumulation=1,
        ulysses_sequence_parallel_size=1,
        device_name="cpu",
        _forward_micro_batch=lambda mb, **kw: fake_forward(None, mb, **kw),
        _optimizer_step=lambda: torch.tensor(1.0),
    )
    return worker, optimizer


def _make_data():
    return _FakeDataProto(
        batch={
            "input_ids": torch.arange(8).reshape(2, 4).long(),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "position_ids": torch.zeros(2, 4, dtype=torch.long),
            "responses": torch.zeros(2, 2, dtype=torch.long),
            "response_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            "teacher_input_ids": torch.arange(10).reshape(2, 5).long() + 100,
            "teacher_attention_mask": torch.ones(2, 5, dtype=torch.long),
            "teacher_position_ids": torch.zeros(2, 5, dtype=torch.long),
            "teacher_response": torch.arange(6).reshape(2, 3).long() + 200,
        },
        meta_info={
            "micro_batch_size": 2,
            "use_dynamic_bsz": False,
            "max_token_len": 1024,
        },
    )


def test_update_step_returns_required_metrics(monkeypatch):
    from easyopd.methods.gad.critic_update import update_critic_step

    worker, optimizer = _make_worker(monkeypatch)
    out = update_critic_step(worker, _make_data())

    metrics = out.meta_info["metrics"]
    for key in (
        "critic/d_loss",
        "critic/d_acc",
        "critic/student_value_mean",
        "critic/teacher_value_mean",
        "critic/grad_norm",
    ):
        assert key in metrics, f"missing metric {key} in {metrics}"


def test_update_step_validates_data_contract(monkeypatch):
    from easyopd.methods.gad.critic_update import update_critic_step
    from easyopd.methods.gad.data_contract import GADBatchContractError

    worker, _ = _make_worker(monkeypatch)
    bad = _make_data()
    del bad.batch["teacher_input_ids"]
    with pytest.raises(GADBatchContractError):
        update_critic_step(worker, bad)
```

- [ ] **Step 2: Run and confirm tests fail**

Run: `pytest tests/easyopd/gad/test_critic_update_contract.py -q`
Expected: 2 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `critic_update.py`**

Create `easyopd/methods/gad/critic_update.py`:

```python
"""GAD critic update loop.

Drop-in replacement for the body of `DataParallelPPOCritic.update_critic`
when `gad.enable=true`. The body runs the critic forward twice per
micro-batch (once on student, once on teacher), computes the Bradley-
Terry pairwise loss, accumulates gradients, takes an optimizer step,
and returns a DataProto carrying per-step metrics.

Validation: on every call we run `validate_gad_batch` against the
data's batch keys. This is cheap (a dict membership check) and the
single place where the GAD data contract is enforced at runtime.
"""

from __future__ import annotations

from typing import Any

from easyopd.methods.gad.core import (
    compute_discriminator_loss,
    discriminator_accuracy,
    summed_reward,
)
from easyopd.methods.gad.data_contract import validate_gad_batch


def _append(metrics: dict, **kvs) -> None:
    for k, v in kvs.items():
        metrics.setdefault(k, []).append(v)


def _reduce(metrics: dict) -> dict:
    return {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in metrics.items()}


def update_critic_step(worker: Any, data: Any):
    """Run one critic update step under GAD's discriminator semantics.

    Args:
        worker: a verl DataParallelPPOCritic instance. We use:
            * `_forward_micro_batch(micro_batch, compute_teacher=...)`
            * `_optimizer_step()`
            * `gradient_accumulation`, `config`, `critic_module`, `critic_optimizer`
        data: a DataProto whose `.batch` includes both student and teacher tensors.

    Returns:
        A DataProto with `meta_info["metrics"]` containing
        `critic/d_loss`, `critic/d_acc`, `critic/student_value_mean`,
        `critic/teacher_value_mean`, `critic/grad_norm`.
    """
    validate_gad_batch(data.batch)

    # Import lazily so the GAD package stays importable without verl.
    from verl.protocol import DataProto  # noqa: WPS433

    worker.critic_module.train()
    worker.critic_optimizer.zero_grad()

    use_dynamic_bsz = bool(getattr(worker.config, "use_dynamic_bsz", False))
    ppo_mini = int(getattr(worker.config, "ppo_mini_batch_size", 1))

    metrics: dict[str, list[float]] = {}

    micro_batches = data.split(int(getattr(worker.config, "ppo_micro_batch_size_per_gpu", ppo_mini)))

    for micro in micro_batches:
        # Re-use the micro-batch dict directly; the forward adapter will copy.
        mb = {**micro.batch, **micro.non_tensor_batch}

        student_vpreds = worker._forward_micro_batch(mb, compute_teacher=False)
        teacher_vpreds = worker._forward_micro_batch(mb, compute_teacher=True)

        # response_mask is sliced to the last `response_length` tokens of the relevant attention_mask.
        # Student: from student attention_mask. Teacher: from teacher_attention_mask.
        s_len = student_vpreds.shape[-1]
        t_len = teacher_vpreds.shape[-1]
        response_mask = mb["attention_mask"][:, -s_len:].to(student_vpreds.dtype)
        teacher_mask = mb["teacher_attention_mask"][:, -t_len:].to(teacher_vpreds.dtype)

        d_loss = compute_discriminator_loss(
            student_vpreds=student_vpreds,
            teacher_vpreds=teacher_vpreds,
            response_mask=response_mask,
            teacher_response_mask=teacher_mask,
        )
        d_acc = discriminator_accuracy(
            student_vpreds=student_vpreds,
            teacher_vpreds=teacher_vpreds,
            response_mask=response_mask,
            teacher_response_mask=teacher_mask,
        )
        s_mean = summed_reward(student_vpreds, response_mask).mean().item()
        t_mean = summed_reward(teacher_vpreds, teacher_mask).mean().item()

        if use_dynamic_bsz:
            loss = d_loss * (student_vpreds.shape[0] / max(ppo_mini, 1))
        else:
            loss = d_loss / max(int(getattr(worker, "gradient_accumulation", 1)), 1)

        loss.backward()

        _append(
            metrics,
            **{
                "critic/d_loss": d_loss.detach().item(),
                "critic/d_acc": float(d_acc),
                "critic/student_value_mean": s_mean,
                "critic/teacher_value_mean": t_mean,
            },
        )

    grad_norm = worker._optimizer_step()
    _append(metrics, **{"critic/grad_norm": float(grad_norm) if not hasattr(grad_norm, "item") else grad_norm.item()})

    out = DataProto(batch=data.batch, meta_info={"metrics": _reduce(metrics)})
    return out
```

- [ ] **Step 4: Run the tests and confirm they pass**

The test uses a `_FakeDataProto` rather than the real verl DataProto. Because `update_critic_step` imports the real DataProto for its return value, we must patch it. Update the test file to add a fixture that monkeypatches the import:

Append to `tests/easyopd/gad/test_critic_update_contract.py`:

```python
@pytest.fixture(autouse=True)
def _stub_dataproto(monkeypatch):
    """Replace verl.protocol.DataProto with _FakeDataProto for the contract test."""
    import sys
    import types

    fake_module = types.ModuleType("verl.protocol")
    fake_module.DataProto = _FakeDataProto  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl.protocol", fake_module)
    yield
```

Re-run: `pytest tests/easyopd/gad/test_critic_update_contract.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add easyopd/methods/gad/critic_update.py \
        tests/easyopd/gad/test_critic_update_contract.py
git commit -m "feat(gad): add update_critic_step (BT-loss discriminator update)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire `[EasyOPD:GAD]` hooks into `verl/workers/critic/dp_critic.py`

**Files:**
- Modify: `verl/workers/critic/dp_critic.py` (3 wraps in 2 methods)

There are exactly **3 comment-wrapped insertion points** in this file. Make every edit small, additive, and bracketed with `# ============ [EasyOPD:GAD] ... # ============ [EasyOPD:GAD] End ============` so a grep can locate them.

- [ ] **Step 1: Add the `compute_teacher` kwarg + remap to `_forward_micro_batch` signature**

Find the line at `verl/workers/critic/dp_critic.py:57`:

```python
    def _forward_micro_batch(self, micro_batch):
```

Replace with:

```python
    def _forward_micro_batch(self, micro_batch, *, compute_teacher: bool = False):
        # ============ [EasyOPD:GAD] Swap input keys when scoring teacher ============
        from easyopd.methods.gad.config import is_gad_enabled

        _gad_active = is_gad_enabled(self.config)
        if _gad_active and compute_teacher:
            from easyopd.methods.gad.critic_forward import remap_to_teacher

            micro_batch = remap_to_teacher(micro_batch)
        # ============ [EasyOPD:GAD] End ============
```

- [ ] **Step 2: Find the `_forward_micro_batch` return line and wrap the last-token reduction**

The method ends with `return values` (search for the FIRST `return values` in the file, which terminates `_forward_micro_batch`). In the current file this is line ~149.

Before that `return values`, locate the existing assignment to `response_mask` (the line `response_mask = attention_mask[:, -response_length:]` near line ~121 in the existing file, or compute it just above the return if not present).

Insert just before the final `return values`:

```python
            # ============ [EasyOPD:GAD] Reduce to last-token-only score ============
            if _gad_active:
                from easyopd.methods.gad.core import last_token_only

                _response_mask_for_gad = attention_mask[:, -response_length:].to(values.dtype)
                values = last_token_only(values, _response_mask_for_gad)
            # ============ [EasyOPD:GAD] End ============
```

If the original code already has a local `response_mask` in scope, reuse it instead of recomputing `_response_mask_for_gad`. The intent is identical.

- [ ] **Step 3: Add the dispatch wrap at the top of `update_critic`**

Find `verl/workers/critic/dp_critic.py:197`:

```python
    def update_critic(self, data: DataProto):
```

Insert immediately after the def line (before any existing body):

```python
    def update_critic(self, data: DataProto):
        # ============ [EasyOPD:GAD] Discriminator-as-critic update ============
        from easyopd.methods.gad.config import is_gad_enabled

        if is_gad_enabled(self.config):
            from easyopd.methods.gad.critic_update import update_critic_step

            return update_critic_step(self, data)
        # ============ [EasyOPD:GAD] End ============
```

(Leave the original body of `update_critic` untouched after the wrap.)

- [ ] **Step 4: Verify the import path resolves with no verl runtime needed**

Run:
```bash
python -c "from easyopd.methods.gad.config import is_gad_enabled; print(is_gad_enabled(None))"
```
Expected: `False`.

- [ ] **Step 5: Verify wrap count grep-wise**

Run:
```bash
grep -c "# ============ \[EasyOPD:GAD\]" verl/workers/critic/dp_critic.py
```
Expected: `6` (3 wraps × 2 lines each: opening + `End`).

- [ ] **Step 6: Re-run the full GAD test suite to confirm nothing regressed**

Run: `pytest tests/easyopd/gad/ -q`
Expected: All previously-passing tests still pass (drift tests will be added in Task 9).

- [ ] **Step 7: Commit**

```bash
git add verl/workers/critic/dp_critic.py
git commit -m "feat(verl): add [EasyOPD:GAD] hooks in dp_critic.py

Three comment-wrapped insertion points:
  - _forward_micro_batch: accept compute_teacher kwarg, remap_to_teacher.
  - _forward_micro_batch: reduce to last-token-only when GAD is on.
  - update_critic: dispatch to easyopd.methods.gad.critic_update.

No verl original code removed; gad.enable=false path is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Add drift / no-actor-changes / smoke tests

**Files:**
- Create: `tests/easyopd/gad/test_no_drift.py`
- Create: `tests/easyopd/gad/test_no_actor_changes.py`
- Create: `tests/easyopd/gad/test_config_smoke.py`

- [ ] **Step 1: Write the drift test**

Create `tests/easyopd/gad/test_no_drift.py`:

```python
"""Verify the [EasyOPD:GAD] integration points remain exactly where we placed them."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _count_marker_lines(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return sum(1 for line in text.splitlines() if "# ============ [EasyOPD:GAD]" in line)


def test_dp_critic_has_exactly_three_wraps():
    # 3 wraps = 3 opening lines + 3 End lines = 6 marker lines.
    n = _count_marker_lines(REPO_ROOT / "verl/workers/critic/dp_critic.py")
    assert n == 6, f"expected 6 [EasyOPD:GAD] marker lines in dp_critic.py, found {n}"


def test_no_marker_in_ray_trainer():
    n = _count_marker_lines(REPO_ROOT / "verl/trainer/ppo/ray_trainer.py")
    assert n == 0, f"expected 0 [EasyOPD:GAD] markers in ray_trainer.py (plan honored), found {n}"
```

- [ ] **Step 2: Write the no-actor-changes test**

Create `tests/easyopd/gad/test_no_actor_changes.py`:

```python
"""GAD must not modify the actor file."""

from pathlib import Path


def test_dp_actor_has_no_gad_markers():
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "verl/workers/actor/dp_actor.py"
    text = target.read_text(encoding="utf-8")
    assert "[EasyOPD:GAD]" not in text, "GAD must not modify dp_actor.py"
```

- [ ] **Step 3: Run the new tests, expect them to pass**

Run: `pytest tests/easyopd/gad/test_no_drift.py tests/easyopd/gad/test_no_actor_changes.py -q`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/easyopd/gad/test_no_drift.py tests/easyopd/gad/test_no_actor_changes.py
git commit -m "test(gad): assert verl integration markers and actor untouched

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Add Hydra config

**Files:**
- Create: `easyopd/config/gad/base.yaml`
- Create: `tests/easyopd/gad/test_config_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/easyopd/gad/test_config_smoke.py`:

```python
"""Smoke tests for the GAD Hydra base config."""

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
GAD_BASE = REPO_ROOT / "easyopd/config/gad/base.yaml"


def test_base_yaml_loads():
    cfg = OmegaConf.load(GAD_BASE)
    assert cfg is not None


def test_base_yaml_exposes_gad_node():
    cfg = OmegaConf.load(GAD_BASE)
    assert "gad" in cfg
    assert "enable" in cfg.gad
    assert "discriminator_init_path" in cfg.gad


def test_is_gad_enabled_reads_yaml():
    from easyopd.methods.gad.config import is_gad_enabled

    cfg = OmegaConf.load(GAD_BASE)
    # User must override discriminator_init_path; enabled=true in base.
    assert is_gad_enabled(cfg) is True
```

- [ ] **Step 2: Run and expect failure (missing yaml)**

Run: `pytest tests/easyopd/gad/test_config_smoke.py -q`
Expected: failures referencing missing `easyopd/config/gad/base.yaml`.

- [ ] **Step 3: Create the YAML**

Create `easyopd/config/gad/base.yaml`:

```yaml
# Generative Adversarial Distillation (GAD) base configuration.
#
# This Hydra fragment is intended to compose on top of verl's PPO defaults.
# The user MUST override `gad.discriminator_init_path` (a path to a pretrained
# discriminator checkpoint, e.g. a warmup-stage SFT model).
#
# See docs/algo/gad.md for the end-to-end recipe.

# yamllint disable rule:line-length

gad:
  enable: true
  discriminator_init_path: ???   # REQUIRED — pretrained discriminator checkpoint
  metrics_prefix: gad

# GAD repurposes the critic as the reward source; no separate RM.
reward_model:
  enable: false

# critic.model.path is left to the user — they will typically set it to
# `${gad.discriminator_init_path}` in their launch script or composed config.
```

- [ ] **Step 4: Run smoke tests; expect them to pass**

Run: `pytest tests/easyopd/gad/test_config_smoke.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add easyopd/config/gad/base.yaml tests/easyopd/gad/test_config_smoke.py
git commit -m "feat(gad): add Hydra base config (easyopd/config/gad/base.yaml)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Add launch entrypoint + dry-run test

**Files:**
- Create: `examples/gad_trainer/README.md`
- Create: `examples/gad_trainer/train_gad.sh`
- Create: `tests/easyopd/gad/test_entrypoints.py`

- [ ] **Step 1: Write the failing entrypoint test**

Create `tests/easyopd/gad/test_entrypoints.py`:

```python
"""Existence + shape smoke tests for examples/gad_trainer/."""

import os
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_DIR = REPO_ROOT / "examples/gad_trainer"


def test_train_script_exists_and_is_executable():
    script = TRAINER_DIR / "train_gad.sh"
    assert script.exists(), "examples/gad_trainer/train_gad.sh missing"
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR, "train_gad.sh must be executable"


def test_train_script_mentions_required_overrides():
    script = (TRAINER_DIR / "train_gad.sh").read_text(encoding="utf-8")
    # The user MUST set discriminator_init_path; the script should make that obvious.
    assert "discriminator_init_path" in script


def test_readme_exists():
    readme = TRAINER_DIR / "README.md"
    assert readme.exists(), "examples/gad_trainer/README.md missing"
    text = readme.read_text(encoding="utf-8")
    assert "discriminator_init_path" in text
    assert "teacher_response" in text
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/easyopd/gad/test_entrypoints.py -q`
Expected: 3 failures (paths missing).

- [ ] **Step 3: Create `train_gad.sh`**

Create `examples/gad_trainer/train_gad.sh`:

```bash
#!/usr/bin/env bash
# Launch GAD adversarial-stage training.
#
# Required overrides (no defaults — fail fast if missing):
#   gad.discriminator_init_path=<path/to/pretrained/discriminator>
#   data.train_files=<path/to/parquet/with/teacher_response>
#   actor_rollout_ref.model.path=<path/to/student>
#   critic.model.path=<path/to/critic>     # typically the discriminator checkpoint
#
# Example:
#   bash examples/gad_trainer/train_gad.sh \
#     gad.discriminator_init_path=/data/disc.ckpt \
#     data.train_files=/data/lmsys_gpt5_chat.parquet \
#     actor_rollout_ref.model.path=/data/qwen2.5-7b-instruct \
#     critic.model.path=/data/disc.ckpt
#
# Dry-run (Hydra resolves the config then exits before launching workers):
#   bash examples/gad_trainer/train_gad.sh ... +hydra.mode=run +dry_run=true

set -euo pipefail

CONFIG_DIR="$(cd "$(dirname "$0")/../../easyopd/config/gad" && pwd)"

python -m verl.trainer.main_ppo \
  --config-path "${CONFIG_DIR}" \
  --config-name base \
  "$@"
```

Then make it executable:
```bash
chmod +x examples/gad_trainer/train_gad.sh
```

- [ ] **Step 4: Create `examples/gad_trainer/README.md`**

Create `examples/gad_trainer/README.md`:

```markdown
# GAD trainer entrypoint

This directory holds the shell launcher for the GAD adversarial-training
stage. It composes the base config at `easyopd/config/gad/base.yaml`
with user-supplied overrides.

## Data preparation

GAD expects each training row to provide four teacher-side tensor
fields (already tokenized to match the discriminator):

- `teacher_input_ids` — full critic input (prompt + teacher response)
- `teacher_attention_mask`
- `teacher_position_ids`
- `teacher_response` — just the teacher response token ids (length-only marker)

We do not ship a data-preparation script. To reproduce the paper:

1. Download the LMSYS-Chat-GPT-5-Chat-Response dataset from
   https://huggingface.co/datasets/ytz20/LMSYS-Chat-GPT-5-Chat-Response.
2. Convert to parquet using the upstream tool at
   `microsoft/LMOps/gad/tools/export_lmsys_parquet.py`.
3. Point `data.train_files` at the resulting parquet path.

## Discriminator checkpoint

The discriminator is a separately trained model (typically the result
of SFT on teacher responses, i.e. the upstream "warmup" stage). EasyOPD
does NOT provide warmup tooling — bring your own checkpoint.

Set `gad.discriminator_init_path=<path>` and `critic.model.path=${gad.discriminator_init_path}`.

## Launching

```bash
bash examples/gad_trainer/train_gad.sh \
  gad.discriminator_init_path=/data/disc.ckpt \
  data.train_files=/data/lmsys_gpt5_chat.parquet \
  actor_rollout_ref.model.path=/data/qwen2.5-7b-instruct \
  critic.model.path=/data/disc.ckpt
```
```

- [ ] **Step 5: Run entrypoint tests; expect them to pass**

Run: `pytest tests/easyopd/gad/test_entrypoints.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add examples/gad_trainer/README.md examples/gad_trainer/train_gad.sh \
        tests/easyopd/gad/test_entrypoints.py
git commit -m "feat(gad): add examples/gad_trainer/ launch script and README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Method docs + toctree

**Files:**
- Create: `docs/algo/gad.md`
- Modify: `docs/index.rst` (add toctree entry)

- [ ] **Step 1: Write `docs/algo/gad.md`**

Create `docs/algo/gad.md`:

```markdown
# GAD: Generative Adversarial Distillation

**Paper:** [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)
**Source code:** `easyopd/methods/gad/`
**Launch:** `examples/gad_trainer/train_gad.sh`

## TL;DR

GAD repurposes verl's PPO critic as a Bradley-Terry discriminator over
student vs teacher responses. The discriminator emits a sequence-level
score (placed at the last response token) which the standard PPO
advantage path consumes as a token-level reward. The actor update is
unchanged.

## How it sits inside verl

| Component | Verl behavior | GAD override |
|-----------|---------------|--------------|
| Rollout | Generates student responses | Unchanged |
| Critic forward (`_forward_micro_batch`) | Per-token value head | Same forward; reduced to last-token-only when `gad.enable=true`; can also forward teacher inputs via `compute_teacher=True` |
| `compute_values` | Returns per-token values | Returns last-token-only score (via the forward override) |
| Advantage computation | GAE/GRPO from token-level rewards | Unchanged; token_level_scores = critic values |
| `update_critic` | MSE value regression | Replaced with Bradley-Terry pairwise loss against teacher responses |
| Actor update | Standard PPO | Unchanged |

All overrides live in `easyopd/methods/gad/`. The only verl file with
EasyOPD-side edits is `verl/workers/critic/dp_critic.py` (3 comment-
wrapped insertion points, ~19 lines).

## Data preparation

Each training sample must provide four teacher-side tensor batch keys:

| Key | Shape | Meaning |
|-----|-------|---------|
| `teacher_input_ids` | `[B, T_full_t]` | Full critic input (prompt + teacher response) |
| `teacher_attention_mask` | `[B, T_full_t]` | Attention mask for the full input |
| `teacher_position_ids` | `[B, T_full_t]` | Position ids for the full input |
| `teacher_response` | `[B, T_resp_t]` | Teacher response tokens only (used to derive response_length) |

These are tensor batch entries and survive verl's `_get_gen_batch` pop
list because that list only contains the three student-side keys.

To reproduce the paper:

1. Pull the LMSYS-Chat-GPT-5-Chat-Response dataset from HuggingFace
   (`ytz20/LMSYS-Chat-GPT-5-Chat-Response`).
2. Convert with the upstream `microsoft/LMOps/gad/tools/export_lmsys_parquet.py`.
3. Point `data.train_files` at the resulting parquet.

## Discriminator setup

Bring your own discriminator checkpoint (typically an SFT model trained
on teacher responses — i.e. the upstream "warmup" stage, which is out
of scope for this EasyOPD integration). Set:

```yaml
gad:
  enable: true
  discriminator_init_path: /path/to/discriminator.ckpt
critic:
  model:
    path: ${gad.discriminator_init_path}
```

## Reference

```bibtex
@article{ye2025blackboxonpolicydistillationlarge,
  title={Black-Box On-Policy Distillation of Large Language Models},
  author={Tianzhu Ye and Li Dong and Zewen Chi and Xun Wu and Shaohan Huang and Furu Wei},
  journal={arXiv preprint arXiv:2511.10643},
  year={2025},
  url={https://arxiv.org/abs/2511.10643}
}
```
```

- [ ] **Step 2: Wire into `docs/index.rst`**

Find the block listing other `algo/*.md` entries (search for `algo/opd.md` — line ~85):

```bash
grep -n "algo/opd.md" docs/index.rst
```

Add a new line directly after `algo/opd.md`:

```rst
   algo/gad.md
```

The final line ordering will be:

```rst
   algo/opd.md
   algo/gad.md
```

- [ ] **Step 3: Sanity-check the docs index**

Run:
```bash
grep -n "algo/gad.md" docs/index.rst
```
Expected: exactly one line.

- [ ] **Step 4: Commit**

```bash
git add docs/algo/gad.md docs/index.rst
git commit -m "docs(gad): add method documentation page and toctree entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Final verification pass

**Files:**
- (none modified; verification only)

- [ ] **Step 1: Run the full GAD test suite**

Run: `pytest tests/easyopd/gad/ -q`
Expected: all tests pass (12 files, ~30+ test cases).

- [ ] **Step 2: Confirm zero impact on the wider test suite when GAD is off**

Run a representative subset of existing tests that exercise the critic without GAD:

```bash
pytest tests/workers/ -q -k "critic" 2>&1 | tail -20
```

Expected: any pre-existing pass/fail counts are unchanged by our edits (no NEW failures attributable to `[EasyOPD:GAD]` hooks).

If you cannot run the workers tests in your environment (they may require CUDA), at minimum confirm:

```bash
python -c "
from omegaconf import OmegaConf
from easyopd.methods.gad.config import is_gad_enabled
cfg = OmegaConf.create({'foo': 1})
assert is_gad_enabled(cfg) is False, 'GAD must be off for unrelated cfg'
print('OK: is_gad_enabled returns False for non-GAD cfg')
"
```

- [ ] **Step 3: grep the codebase for stray `[EasyOPD:GAD]` markers**

Run:
```bash
grep -rln "\[EasyOPD:GAD\]" verl/ docs/ examples/ easyopd/
```

Expected output:
```
easyopd/methods/gad/__init__.py
easyopd/methods/gad/README.md
verl/workers/critic/dp_critic.py
tests/easyopd/gad/test_no_drift.py
tests/easyopd/gad/test_no_actor_changes.py
```

(The `tests/` files mention the string in assertions.)

No other file should appear in the list. If extra files appear, investigate before declaring the task done.

- [ ] **Step 4: Inspect the diff against `main`**

```bash
git diff --stat main...HEAD
```

Expected new-file count:
- `easyopd/methods/gad/`: 6 files (`__init__.py`, `core.py`, `data_contract.py`, `config.py`, `critic_forward.py`, `critic_update.py`, plus `README.md` = 7)
- `easyopd/config/gad/base.yaml`: 1 file
- `examples/gad_trainer/`: 2 files
- `docs/algo/gad.md`: 1 file
- `tests/easyopd/gad/`: 12 files
- spec doc: already committed (Task pre-cursor)
- plan doc: this file
- `verl/workers/critic/dp_critic.py`: modified (~19 lines added)
- `docs/index.rst`: modified (1 line added)

Total: ~25 new files, 2 modified, plus the spec + plan committed before Task 1.

- [ ] **Step 5: Final commit (only if there are uncommitted leftovers)**

If `git status` is clean, no commit is needed.

If there are leftover changes (e.g. a stray formatting artifact), make one final commit:

```bash
git add -A
git commit -m "chore(gad): final cleanup of GAD migration artifacts

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Out-of-scope work (explicit non-deliverables)

The following are intentionally NOT covered by this plan, per the spec
non-goals section and the implementer's clarifying-question answers:

- SeqKD baseline (would require an SFT-style loop replacement).
- Warmup stage (the upstream Stage 1 SFT used to bootstrap the discriminator).
- Eval-only / generation-only mode.
- Teacher data preparation script (referenced in `docs/algo/gad.md` instead).
- Automatic discriminator-checkpoint download or warmup pretraining.
- Multi-GPU or end-to-end GPU validation. The CPU contract suite verifies
  structure, not numerical equivalence to the upstream gad fork.
- Upstream PR back to `verl-project/verl` (would require duplicate-work
  checks per `AGENTS.md`).
