# Contributing a New Method to EasyOPD

EasyOPD is designed so that adding a new On-Policy Distillation (OPD) method
is a **5-step**, **single-PR** task. This guide walks through every file you
need to touch, with concrete code snippets pulled from the simplest existing
method (`gkd` — pure `LossHook`, no rollout/reward/sidecar). After reading
this page you should be able to land a working baseline for "method X" in
under a day.

---

## Architecture in 30 seconds

EasyOPD adds a thin **registry + hook protocol** layer on top of `verl`:

```text
┌────────────────────────────────────────────────────────────┐
│ easyopd/registry.py        @register_method("xxx")         │
│ easyopd/hooks.py           5 Protocol classes              │
│ easyopd/hook_dispatch.py   capability-based dispatch       │
│ scripts/test_all_methods   regression matrix (12 methods)  │
└────────────────────────────────────────────────────────────┘
```

The 5 hook Protocols (any subset is fine — duck-typed):

| Hook | Implement when… |
|---|---|
| `LossHook` | Your method has a custom distillation loss (KL/JSD/CE variant). |
| `RolloutHook` | You need to mutate the on-policy generation pipeline. |
| `RewardHook` | You compute rewards differently from verl's defaults. |
| `AlignmentHook` | Cross-tokenizer / sub-vocab projection is needed. |
| `TeacherSidecarHook` | You run a live vLLM/sglang teacher inside Ray. |

For methods that **don't** plug into the actor (e.g. critic-as-discriminator
or advantage-estimator-only), see [Step 2 sidebar](#sidebar-non-actor-methods).

---

## The 5 steps

### Step 1 — Register metadata

Create `easyopd/methods/<your_method>/__init__.py` and decorate a metadata
class with `@register_method`. Required attributes: `name`, `description`,
`paper_url`, `verl_modified_files`, `capabilities`.

**Concrete example — `gkd`:**

```python
# easyopd/methods/gkd/__init__.py
from easyopd.registry import register_method


@register_method("gkd")
class GKDMethod:
    """GKD: Generalized Knowledge Distillation."""

    # --- Required metadata ---
    name = "gkd"
    description = (
        "GKD: Generalized Knowledge Distillation. "
        "On-policy distillation using generalized JSD with dense teacher feedback."
    )
    paper_url = "https://arxiv.org/abs/2306.13649"
    code_url = "https://github.com/shawnli/on-policy-distillation"  # optional

    # Files modified in verl/ (must use the comment-bracketed marker
    # convention — see "verl-side modifications" below).
    verl_modified_files = [
        "verl/trainer/distillation/losses.py",
        "verl/workers/config/distillation.py",
    ]

    # Which hook capabilities this method exposes.
    capabilities = ("loss",)
```

> Place core algorithm code in `easyopd/methods/<your_method>/core.py`
> and re-export the public API from `__init__.py`.

### Step 2 — Implement the relevant Hook(s)

Create `easyopd/methods/<your_method>/hooks.py`. Each Protocol is duck-typed;
you do **not** need to inherit from anything.

**Concrete example — `gkd` `LossHook`:**

```python
# easyopd/methods/gkd/hooks.py
from typing import Any
import torch
from easyopd.hooks import Config, Metrics


class GKDLossHook:                         # ← duck-typed, no inheritance
    """Implements the LossHook Protocol for GKD."""

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        from easyopd.methods.gkd.core import gkd_loss

        beta = (config.get("beta", 0.5) if isinstance(config, dict)
                else getattr(config, "beta", 0.5))
        temperature = kwargs.get("temperature", 1.0)
        loss, metrics = gkd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            beta=beta,
            temperature=temperature,
        )
        return loss, metrics
```

For other Protocols see `easyopd/hooks.py` — each has a single primary method
(`generate` / `compute_reward` / `align_logits` / `start_teacher` etc.) and a
short docstring describing the contract.

#### Sidebar — non-actor methods

If your method doesn't plug into the actor's loss/rollout/reward path
(e.g. **GAD** repurposes the critic as a discriminator; **lightning_opd**
implements an advantage estimator routed through verl directly), add the
method name to `NON_ACTOR_HOOK_METHODS` in `scripts/test_all_methods.py`:

```python
# scripts/test_all_methods.py
NON_ACTOR_HOOK_METHODS = {"gad", "lightning_opd", "your_new_method"}
```

This tells the regression matrix to skip the actor-hook validation for
that method while keeping every other check (registry, config, dataset,
verl marker discipline) active.

### Step 3 — Expose a default config

Create `easyopd/config/<your_method>.yaml`. **It must be self-contained**
(EasyOPD's `_load_config` does not perform Hydra `defaults`-style include
merging at the framework level — Hydra include only happens inside the
launcher script). Mirror the sectioning convention used by `gkd.yaml` /
`simple.yaml`:

```yaml
# easyopd/config/<your_method>.yaml
method:
  name: <your_method>
  description: "..."

model:
  student_model_path: "..."
  teacher_model_path: "..."

training:
  adv_estimator: grpo
  loss_agg_mode: token-mean
  actor_lr: 5e-6
  train_batch_size: 64
  ppo_mini_batch_size: 64
  total_epochs: 1

distillation:
  enabled: true
  n_gpus_per_node: 8
  nnodes: 1

rollout:
  max_prompt_length: 1024
  max_response_length: 512
  temperature: 1.0
  n: 1
  gpu_memory_utilization: 0.8

data:
  dataset: "openai/gsm8k"
  dataset_split: "train"
  val_split: "test"
  prompt_template: "math_qa"
  prompt_key: content
  max_prompt_length: 1024
  truncation: right
```

If your method needs many low-level verl-hydra fragments (multi-stage SFT,
provider profiles, etc.), put them in a `easyopd/config/<your_method>/`
subdirectory and have your launch shell compose them with Hydra; the
top-level `<your_method>.yaml` is the **single canonical entry point** for
`EasyOPD.from_hparams("<your_method>")`.

### Step 4 — Register the data recipe

Add a row to `DATASET_RECIPES` in `easyopd/data_provider.py` so that
`auto_resolve_data=True` can download + convert the default HF dataset.
Pick the closest existing `prompt_template` (`math_qa`, `agent`, `code`,
`prompt_only`, `vision_qa`, `raw_chat`); only invent a new template if the
target schema is genuinely incompatible with all existing ones.

```python
# easyopd/data_provider.py
DATASET_RECIPES = {
    # ... existing entries ...
    "your_method": {
        "dataset": "openai/gsm8k",
        "split": "train",
        "val_split": "test",
        "prompt_template": "math_qa",
        # "extra_columns": [...],   # only if your loss needs more
    },
}
```

### Step 5 — Extend the regression matrix

Add your method to **two** locations in `scripts/test_all_methods.py`:

```python
# 1) Whitelist
EXPECTED_METHODS = [
    "g_opd", "gad", "gkd", "lightning_opd", "opcd", "opsa", "ropd",
    "sdpo", "simct", "simple", "sod", "vision_opd",
    "your_method",          # ← new entry
]

# 2) Hook capability map
expected_hooks = {
    # ... existing entries ...
    "your_method": {"loss"},        # subset of {loss, rollout, reward,
                                    #            alignment, teacher_sidecar}
}
```

If your method is non-actor, also append it to `NON_ACTOR_HOOK_METHODS`
(see the [Step 2 sidebar](#sidebar-non-actor-methods)).

Then run:

```bash
python scripts/test_all_methods.py
# Expected: Results: 10 passed, 0 failed
```

If any of the 10 sub-tests fails, the regression report tells you exactly
which contract was violated.

---

## verl-side modifications (mandatory marker convention)

When your method requires changes to files inside `verl/` (registry hooks,
extra config fields, etc.), you **MUST** wrap the change in a comment-bracketed
block so reviewers can locate every cross-package extension at a glance:

```python
# [EasyOPD:GKD] Register the GKD JSD loss mode.
register_distillation_loss("gkd", lambda *a, **kw: gkd_loss(*a, **kw))
# [EasyOPD:GKD] End

# [EasyOPD:GKD] Add gkd_beta to DistillationLossConfig.
@dataclass
class DistillationLossConfig:
    # ... existing fields ...
    gkd_beta: float = 0.5
# [EasyOPD:GKD] End
```

* Use the **uppercase method name** between the colon and the closing bracket.
* Open the block with `# [EasyOPD:NAME]` and close with `# [EasyOPD:NAME] End`.
* List every modified verl file in `verl_modified_files` (Step 1) — the
  regression matrix's `test_verl_files_documented` sub-test enforces this.
* PRs without the markers will be rejected at review.

---

## Checklist before opening a PR

- [ ] `easyopd/methods/<name>/__init__.py` with `@register_method("<name>")`
- [ ] `easyopd/methods/<name>/hooks.py` with at least one Protocol implementation
      (or `<name>` added to `NON_ACTOR_HOOK_METHODS`)
- [ ] `easyopd/config/<name>.yaml` self-contained at the top level
- [ ] Row added to `easyopd/data_provider.py::DATASET_RECIPES`
- [ ] `scripts/test_all_methods.py`: added to `EXPECTED_METHODS` **and**
      `expected_hooks` **and** (if non-actor) `NON_ACTOR_HOOK_METHODS`
- [ ] All `verl/` modifications wrapped in `# [EasyOPD:NAME] ... # [EasyOPD:NAME] End`
- [ ] `python scripts/test_all_methods.py` reports `10 passed, 0 failed`
- [ ] `EasyOPD.from_hparams("<name>", auto_resolve_data=False)` works without errors
- [ ] (Optional) `docs/algo/<name>.md` describing the algorithm

That's the entire integration surface. Welcome aboard.
