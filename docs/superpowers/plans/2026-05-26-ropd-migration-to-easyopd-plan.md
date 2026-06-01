# ROPD Migration To EasyOPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the maintained ROPD mainline from `black-opd` into EasyOPD as a first-class `easyopd.methods.ropd` method with repo-accurate configs, prompts, entrypoints, tests, and minimal required verl integration.

**Architecture:** Keep ROPD library code under `easyopd/methods/ropd/`, user-facing launch and helper docs under `examples/ropd_trainer/`, and method docs under `docs/algo/ropd.md`. Reuse EasyOPD's existing patterns from `simple` and `simct`, but explicitly plan the missing integration seams: reward-manager loading, packaged prompt resources, and any minimal `verl/` hooks needed to make the migrated method actually runnable.

**Tech Stack:** Python 3.10+, Bash, Hydra, pytest, setuptools package data, `importlib.resources`, verl reward manager registry, EasyOPD method packages.

---

## File Structure

### Files to create

- `easyopd/methods/ropd/__init__.py`
- `easyopd/methods/ropd/pipeline.py`
- `easyopd/methods/ropd/prompt_utils.py`
- `easyopd/methods/ropd/prompts.py`
- `easyopd/methods/ropd/teacher_index.py`
- `easyopd/methods/ropd/clients.py`
- `easyopd/methods/ropd/reward_manager.py`
- `easyopd/methods/ropd/judge/__init__.py`
- `easyopd/methods/ropd/judge/circuit_breaker.py`
- `easyopd/methods/ropd/judge/config.py`
- `easyopd/methods/ropd/judge/openai_env.py`
- `easyopd/methods/ropd/judge/provider.py`
- `easyopd/methods/ropd/judge/rate_limit.py`
- `easyopd/methods/ropd/judge/resolver.py`
- `easyopd/methods/ropd/judge/runtime.py`
- `easyopd/methods/ropd/judge/runtime_builder.py`
- `easyopd/methods/ropd/judge/scheduler.py`
- `easyopd/methods/ropd/judge/schema.py`
- `easyopd/methods/ropd/judge/teacher_client.py`
- `easyopd/methods/ropd/utils/__init__.py`
- `easyopd/methods/ropd/utils/eval_package.py`
- `easyopd/methods/ropd/prompts/rubricator.txt`
- `easyopd/methods/ropd/prompts/rubricator_cn.txt`
- `easyopd/methods/ropd/prompts/verifier.txt`
- `easyopd/methods/ropd/prompts/verifier_cn.txt`
- `easyopd/methods/ropd/prompts/verifier_skywork.txt`
- `easyopd/config/ropd/base.yaml`
- `easyopd/config/ropd/judge.yaml`
- `easyopd/config/ropd/sft.yaml`
- `examples/ropd_trainer/README.md`
- `examples/ropd_trainer/train_ropd.sh`
- `docs/algo/ropd.md`
- `tests/easyopd/ropd/__init__.py`
- `tests/easyopd/ropd/test_pipeline.py`
- `tests/easyopd/ropd/test_reward_manager.py`
- `tests/easyopd/ropd/test_clients.py`
- `tests/easyopd/ropd/test_judge_config.py`
- `tests/easyopd/ropd/test_judge_provider.py`
- `tests/easyopd/ropd/test_prompt_resources.py`
- `tests/easyopd/ropd/test_entrypoints.py`
- `tests/easyopd/ropd/test_no_legacy_names.py`
- `tests/easyopd/ropd/test_registration.py`
- `tests/easyopd/ropd/test_config_smoke.py`

### Files likely to modify

- `easyopd/methods/__init__.py`
- `pyproject.toml`
- `setup.py`
- `docs/index.rst`
- `verl/workers/reward_manager/__init__.py`
- `verl/trainer/ppo/reward.py`
- `tests/workers/reward_manager/test_registry_on_cpu.py`

### Source files to read during implementation

- `/mnt/d/Area/DL/projects/research/black-opd/algo/pipeline.py`
- `/mnt/d/Area/DL/projects/research/black-opd/algo/prompt_utils.py`
- `/mnt/d/Area/DL/projects/research/black-opd/algo/prompts.py`
- `/mnt/d/Area/DL/projects/research/black-opd/algo/teacher_index.py`
- `/mnt/d/Area/DL/projects/research/black-opd/algo/clients.py`
- `/mnt/d/Area/DL/projects/research/black-opd/algo/reward_manager.py`
- `/mnt/d/Area/DL/projects/research/black-opd/algo/judge/*.py`
- `/mnt/d/Area/DL/projects/research/black-opd/algo/utils/eval_package.py`
- `/mnt/d/Area/DL/projects/research/black-opd/training/ppo/train_ropd.sh`
- `/mnt/d/Area/DL/projects/research/black-opd/tests/algo/*.py`
- `/mnt/d/Area/DL/projects/research/black-opd/tests/training/test_shared_rubrics_entrypoints_on_cpu.py`

### Design constraints locked in by this plan

- Only the maintained ROPD mainline migrates. No ablation directories, no `shared_rubrics` compatibility surface, no `BLACK_OPD_*` env alias support.
- The target-side public naming is `ropd` only. Any `shared-rubrics`, `shared_rubrics`, or `black_opd` strings that remain must exist only in source comments/tests that verify they are absent from the new surface.
- Prompt templates become package data under `easyopd/methods/ropd/prompts/` and must load through `importlib.resources`.
- The plan assumes the current EasyOPD repo does **not** already have black-opd's custom `reward_manager.source=importlib` config system, so ROPD must be connected through a target-side loading path that exists in this repo or is added as minimal new integration.

### Decision to make explicit during execution

- Use one of these two reward-manager integration paths and document the choice in Task 3 before proceeding:
```text
Option A (preferred): register a ROPD reward manager into `verl.workers.reward_manager`
under a stable name such as `ropd`, then have `examples/ropd_trainer/train_ropd.sh`
set `reward.reward_manager.name=ropd`.

Option B: add a narrow importlib/file-path reward-manager loader to EasyOPD's current
`verl/trainer/ppo/reward.py`, then point ROPD config at
`easyopd/methods/ropd/reward_manager.py`.
```
- Default recommendation: Option A unless Task 3 finds a hard blocker, because it matches the current EasyOPD reward-manager registry shape and avoids re-porting a larger black-opd-only config subsystem.

## Task 1: Inventory The Real Migration Surface

**Files:**
- Read: `docs/superpowers/specs/2026-05-26-ropd-migration-to-easyopd-design.md`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/algo/pipeline.py`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/algo/prompt_utils.py`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/algo/prompts.py`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/algo/teacher_index.py`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/algo/clients.py`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/algo/reward_manager.py`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/algo/judge/*.py`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/algo/utils/eval_package.py`
- Read: `/mnt/d/Area/DL/projects/research/black-opd/training/ppo/train_ropd.sh`
- Read: `/mnt/d/Area/DL/projects/research/EasyOPD/easyopd/methods/simple/*`
- Read: `/mnt/d/Area/DL/projects/research/EasyOPD/easyopd/methods/simct/*`
- Read: `/mnt/d/Area/DL/projects/research/EasyOPD/verl/trainer/ppo/reward.py`
- Read: `/mnt/d/Area/DL/projects/research/EasyOPD/verl/workers/reward_manager/*`

- [ ] **Step 1: Create a migration inventory note in the implementation branch workspace**

Create a short scratch note outside git tracking, for example:

```text
Task 1 inventory
- Core modules actually needed by ropd mainline:
  pipeline, prompt_utils, prompts, teacher_index, clients, reward_manager,
  judge/*, utils/eval_package
- Not needed in phase 1:
  algo/ablation/*, training/judge/*, experiments/*, data outputs, multi_teacher_index
- Hidden integration seams:
  reward manager loader
  package-data for prompt txt files
  docs toctree entry
```

- [ ] **Step 2: Verify there are no extra source dependencies hidden behind imports**

Run:
```bash
rg -n "^(from algo|import algo)" /mnt/d/Area/DL/projects/research/black-opd/algo/{pipeline.py,prompt_utils.py,prompts.py,teacher_index.py,clients.py,reward_manager.py} /mnt/d/Area/DL/projects/research/black-opd/algo/judge/*.py
```
Expected:
```text
Only the known mainline modules appear; no unexpected dependency on ablation/* or training/*
```

- [ ] **Step 3: Record the exact file list that will migrate**

Expected migrate set:
```text
pipeline.py
prompt_utils.py
prompts.py
teacher_index.py
clients.py
reward_manager.py
judge/{__init__,circuit_breaker,config,openai_env,provider,rate_limit,resolver,runtime,runtime_builder,scheduler,schema,teacher_client}.py
utils/{__init__,eval_package}.py
prompts/{rubricator,rubricator_cn,verifier,verifier_cn,verifier_skywork}.txt
```

- [ ] **Step 4: Confirm the excluded surfaces stay excluded**

Run:
```bash
rg --files /mnt/d/Area/DL/projects/research/black-opd/{algo,training,tests,experiments,data,prompts} | rg "ablation|outputs|wandb|datasets/unified|multi_teacher_index|build_black_opd|validate_shared_rubrics|launch_local_vllm"
```
Expected:
```text
Matches exist in source repo for historical context, but none are added to the target migration file list
```

- [ ] **Step 5: Commit the inventory checkpoint**

```bash
git add docs/superpowers/plans/2026-05-26-ropd-migration-to-easyopd-plan.md
git commit -m "docs: add ropd migration implementation plan"
```

## Task 2: Scaffold The Target ROPD Method Package

**Files:**
- Create: `easyopd/methods/ropd/__init__.py`
- Create: `easyopd/methods/ropd/pipeline.py`
- Create: `easyopd/methods/ropd/prompt_utils.py`
- Create: `easyopd/methods/ropd/prompts.py`
- Create: `easyopd/methods/ropd/teacher_index.py`
- Create: `easyopd/methods/ropd/clients.py`
- Create: `easyopd/methods/ropd/reward_manager.py`
- Create: `easyopd/methods/ropd/judge/*.py`
- Create: `easyopd/methods/ropd/utils/{__init__.py,eval_package.py}`
- Modify: `easyopd/methods/__init__.py`

- [ ] **Step 1: Create the package directories and sentinel files**

Create:
```text
easyopd/methods/ropd/
easyopd/methods/ropd/judge/
easyopd/methods/ropd/utils/
easyopd/methods/ropd/prompts/
```

Add a minimal `__init__.py` patterned after other EasyOPD methods:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ROPDMethod:
    name: str = "ropd"
    description: str = "Rubric-based on-policy distillation mainline migrated from black-opd."


METHOD = ROPDMethod()


def register() -> None:
    from .reward_manager import register_ropd_reward_manager

    register_ropd_reward_manager()


__all__ = ["METHOD", "ROPDMethod", "register"]
```

- [ ] **Step 2: Port `pipeline.py` first and keep names target-side clean**

Port `Group`, `Rollout`, `normalize_raw_prompt`, and `canonicalize_raw_prompt` from source while normalizing module imports to `easyopd.methods.ropd.*`.

Contract to preserve:
```python
def normalize_raw_prompt(raw_prompt: Any) -> RawPrompt: ...
def canonicalize_raw_prompt(raw_prompt: Any) -> str: ...
```

- [ ] **Step 3: Port `prompt_utils.py` and switch all imports to package-local ones**

Implementation rule:
```python
from easyopd.methods.ropd.pipeline import normalize_raw_prompt
```

Do not keep:
```python
from algo.pipeline import normalize_raw_prompt
```

- [ ] **Step 4: Port `teacher_index.py`, `clients.py`, `judge/*`, and `reward_manager.py` without alias branches**

Required transform examples:
```python
from easyopd.methods.ropd.judge.config import ...
from easyopd.methods.ropd.prompts import ...
from easyopd.methods.ropd.teacher_index import ...
```

Remove target-side support for old config-key fallbacks like:
```python
ablation=...
shared_rubrics=...
BLACK_OPD_...
```

Target-side constructor goal:
```python
class ROPDRewardManager(AbstractRewardManager):
    def __init__(..., ropd: ROPDJudgeConfig | dict[str, Any] | None = None, **_: Any) -> None:
        ...
```

- [ ] **Step 5: Port `utils/eval_package.py` only if Task 1 confirmed it is mainline-relevant**

If used by the migrated mainline path, create:
```python
__all__ = ["archive_eval_package", "build_eval_package"]
```

If not used by the ROPD launch or tests, omit this file and update the plan execution log to explain why it was intentionally skipped.

- [ ] **Step 6: Run a compile-only smoke check on the new package skeleton**

Run:
```bash
python -m compileall easyopd/methods/ropd
```
Expected:
```text
Compilation succeeds for all ropd modules
```

- [ ] **Step 7: Run an old-import residue scan on the newly created package**

Run:
```bash
rg -n "from algo|import algo|shared_rubrics|shared-rubrics|BLACK_OPD_|black_opd" easyopd/methods/ropd
```
Expected:
```text
No matches, or only intentional comments/tests that explicitly describe the migration boundary
```

- [ ] **Step 8: Commit the package scaffold**

```bash
git add easyopd/methods/ropd easyopd/methods/__init__.py
git commit -m "feat: scaffold ropd method package"
```

## Task 3: Connect ROPD To EasyOPD's Reward-Manager Loading Path

**Files:**
- Modify: `easyopd/methods/ropd/__init__.py`
- Modify: `easyopd/methods/ropd/reward_manager.py`
- Modify: `verl/workers/reward_manager/__init__.py`
- Modify: `verl/trainer/ppo/reward.py`
- Modify: `tests/workers/reward_manager/test_registry_on_cpu.py`
- Create: `tests/easyopd/ropd/test_registration.py`

- [ ] **Step 1: Choose and document the loading strategy**

Decision note to record at the top of the task scratch log:

```text
Chosen strategy: Option A register `ropd` in the existing reward-manager registry.
Reason: current EasyOPD already resolves reward managers by registry name, while
black-opd's importlib-based reward_manager config subsystem does not exist here.
```

- [ ] **Step 2: Add an idempotent registration helper in `easyopd/methods/ropd/reward_manager.py`**

Target helper shape:
```python
def register_ropd_reward_manager() -> None:
    from verl.workers.reward_manager import register

    @register("ropd")
    class _RegisteredROPDRewardManager(ROPDRewardManager):
        pass
```

If the registry decorator shape makes nested class registration awkward, register the concrete class directly at import time with a small guard.

- [ ] **Step 3: Ensure importing the reward-manager registry also imports ROPD registration**

Prefer a narrow import in:
```python
verl/workers/reward_manager/__init__.py
```

For example:
```python
from easyopd.methods.ropd.reward_manager import register_ropd_reward_manager

register_ropd_reward_manager()
```

Keep this import narrow and avoid pulling optional heavyweight code before needed if possible.

- [ ] **Step 4: Add a direct unit test for registry visibility**

Create:
```python
from verl.workers.reward_manager import get_reward_manager_cls


def test_ropd_reward_manager_registers_under_registry_name():
    cls = get_reward_manager_cls("ropd")
    assert cls.__name__ in {"ROPDRewardManager", "_RegisteredROPDRewardManager"}
```

- [ ] **Step 5: Decide whether `verl/trainer/ppo/reward.py` needs a minimal import trigger**

If importing `verl.workers.reward_manager` is already enough, leave `verl/trainer/ppo/reward.py` untouched.

If not, add a minimal import trigger near other EasyOPD hooks and document it with a succinct comment:
```python
# Ensure EasyOPD custom reward managers are registered before lookup.
import easyopd.methods.ropd  # noqa: F401
```

- [ ] **Step 6: Verify registry lookup works from the current target repo**

Run:
```bash
uv run --no-sync pytest tests/workers/reward_manager/test_registry_on_cpu.py tests/easyopd/ropd/test_registration.py -v
```
Expected:
```text
Registry tests pass and `get_reward_manager_cls("ropd")` resolves
```

- [ ] **Step 7: Commit the reward-manager integration**

```bash
git add easyopd/methods/ropd/reward_manager.py easyopd/methods/ropd/__init__.py verl/workers/reward_manager/__init__.py verl/trainer/ppo/reward.py tests/workers/reward_manager/test_registry_on_cpu.py tests/easyopd/ropd/test_registration.py
git commit -m "feat: register ropd reward manager"
```

## Task 4: Package Prompt Resources Correctly

**Files:**
- Create: `easyopd/methods/ropd/prompts/{rubricator.txt,rubricator_cn.txt,verifier.txt,verifier_cn.txt,verifier_skywork.txt}`
- Modify: `easyopd/methods/ropd/prompts.py`
- Modify: `pyproject.toml`
- Modify: `setup.py`
- Create: `tests/easyopd/ropd/test_prompt_resources.py`

- [ ] **Step 1: Copy the five mainline prompt files into the package resource directory**

Source to target mapping:
```text
/mnt/d/Area/DL/projects/research/black-opd/prompts/custom/rubricator.txt     -> easyopd/methods/ropd/prompts/rubricator.txt
/mnt/d/Area/DL/projects/research/black-opd/prompts/custom/rubricator_cn.txt  -> easyopd/methods/ropd/prompts/rubricator_cn.txt
/mnt/d/Area/DL/projects/research/black-opd/prompts/custom/verifier.txt       -> easyopd/methods/ropd/prompts/verifier.txt
/mnt/d/Area/DL/projects/research/black-opd/prompts/custom/verifier_cn.txt    -> easyopd/methods/ropd/prompts/verifier_cn.txt
/mnt/d/Area/DL/projects/research/black-opd/prompts/custom/verifier_skywork.txt -> easyopd/methods/ropd/prompts/verifier_skywork.txt
```

- [ ] **Step 2: Replace file-path lookup in `prompts.py` with `importlib.resources`**

Target loader shape:
```python
from functools import cache
from importlib import resources


@cache
def _load_template(template_name: str) -> str:
    return resources.files("easyopd.methods.ropd.prompts").joinpath(template_name).read_text(encoding="utf-8")
```

Do not keep:
```python
Path(__file__).resolve().parents[1] / "prompts" / "custom" / template_name
```

- [ ] **Step 3: Add package-data declarations for the new prompt files**

Update `pyproject.toml`:
```toml
[tool.setuptools.package-data]
verl = [
  "version/*",
  "trainer/config/*.yaml",
  "trainer/config/*/*.yaml",
  "experimental/*/config/*.yaml",
]
easyopd = [
  "methods/ropd/prompts/*.txt",
]
```

Update `setup.py`:
```python
package_data={
    "": ["version/*"],
    "verl": [
        "trainer/config/*.yaml",
        "trainer/config/*/*.yaml",
        "experimental/*/config/*.yaml",
    ],
    "easyopd": [
        "methods/ropd/prompts/*.txt",
    ],
}
```

- [ ] **Step 4: Add a focused resource-loading test**

Create a CPU-safe test like:
```python
from importlib import resources


def test_ropd_prompt_resources_load_from_package_data():
    text = resources.files("easyopd.methods.ropd.prompts").joinpath("rubricator.txt").read_text(encoding="utf-8")
    assert text.strip()
```

- [ ] **Step 5: Verify prompt loading and package-data declarations**

Run:
```bash
uv run --no-sync pytest tests/easyopd/ropd/test_prompt_resources.py -v
python -m compileall easyopd/methods/ropd
```
Expected:
```text
Prompt resource test passes and compileall still succeeds
```

- [ ] **Step 6: Commit packaged resources**

```bash
git add easyopd/methods/ropd/prompts easyopd/methods/ropd/prompts.py pyproject.toml setup.py tests/easyopd/ropd/test_prompt_resources.py
git commit -m "feat: package ropd prompt resources"
```

## Task 5: Add ROPD Config Templates Under EasyOPD

**Files:**
- Create: `easyopd/config/ropd/base.yaml`
- Create: `easyopd/config/ropd/judge.yaml`
- Create: `easyopd/config/ropd/sft.yaml`
- Create: `tests/easyopd/ropd/test_config_smoke.py`

- [ ] **Step 1: Create repo-local config templates that match current EasyOPD patterns**

`easyopd/config/ropd/base.yaml` should follow the style of `simple.yaml` and `simct.yaml`: method-local overrides, not a full copy of `verl/trainer/config/ppo_trainer.yaml`.

Minimum intent:
```yaml
defaults:
  - _self_

distillation:
  enabled: true

reward:
  reward_manager:
    name: ropd
```

Add only the ROPD-specific knobs that the launch script consumes.

- [ ] **Step 2: Put judge/provider-resolution settings in `judge.yaml`**

This file should hold the mainline provider-resolution surface for teacher, rubricator, and verifier, including scheduler and quality-gate knobs that are actually used by `ROPDRewardManager`.

Keep target-side names under:
```yaml
reward_model:
  reward_kwargs:
    ropd:
      ...
```

- [ ] **Step 3: Add `sft.yaml` only if the source repo still has an SFT path worth preserving**

If the migrated phase-1 surface still needs `training/sft/train_ropd_sft.sh` semantics, encode the reusable configuration in `easyopd/config/ropd/sft.yaml`.

If not, create the file as a small documented placeholder-free config template that explicitly states it is for follow-on work and is not yet wired to an example launcher.

- [ ] **Step 4: Add a config smoke test that asserts target-side names**

Example test shape:
```python
import yaml
from pathlib import Path


def test_ropd_base_config_uses_ropd_names_only():
    cfg = yaml.safe_load(Path("easyopd/config/ropd/base.yaml").read_text())
    rendered = Path("easyopd/config/ropd/base.yaml").read_text()
    assert "shared_rubrics" not in rendered
    assert "BLACK_OPD_" not in rendered
    assert "black_opd" not in rendered
```

- [ ] **Step 5: Verify config templates are syntactically valid**

Run:
```bash
uv run --no-sync pytest tests/easyopd/ropd/test_config_smoke.py -v
python - <<'PY'
from pathlib import Path
import yaml
for path in Path("easyopd/config/ropd").glob("*.yaml"):
    yaml.safe_load(path.read_text())
    print(path)
PY
```
Expected:
```text
All three YAML files parse successfully
```

- [ ] **Step 6: Commit the config templates**

```bash
git add easyopd/config/ropd tests/easyopd/ropd/test_config_smoke.py
git commit -m "feat: add ropd config templates"
```

## Task 6: Build The Canonical `examples/ropd_trainer` Entrypoint

**Files:**
- Create: `examples/ropd_trainer/train_ropd.sh`
- Create: `examples/ropd_trainer/README.md`
- Create: `tests/easyopd/ropd/test_entrypoints.py`

- [ ] **Step 1: Start from EasyOPD example conventions, not a raw copy of source `train_ropd.sh`**

Use `examples/simple/run_simple.sh` and `examples/simct/run_simct.sh` as the structural template:
```text
- top-level user-adjustable env vars
- derived defaults below the boundary
- one canonical script
- launch via `python3 -m verl.trainer.main_ppo`
```

- [ ] **Step 2: Port only the ROPD-specific environment and override assembly**

Must support:
```text
ROPD_DRYRUN=true
ROPD_SKIP_REPO_DOTENV=true
DATA_ROOT=...
ROPD_* variables only
```

Must not support by default:
```text
BLACK_OPD_*
SHARED_RUBRICS_*
```

- [ ] **Step 3: Implement dry-run mode as a first-class contract**

Dry-run contract:
```bash
ROPD_DRYRUN=true ROPD_SKIP_REPO_DOTENV=true bash examples/ropd_trainer/train_ropd.sh
```

Dry-run output must print:
```text
PROJECT_ROOT=...
Config template(s)=...
DATA_ROOT=...
Student model source=...
Judge provider source=...
Reward manager=ropd
Final python command:
python3 -m verl.trainer.main_ppo ...
```

- [ ] **Step 4: Keep the target-side entrypoint singular**

Only create:
```text
examples/ropd_trainer/train_ropd.sh
```

Do not create:
```text
validate_ropd.sh
launch_ropd_gpu_sweep_tmux.sh
train_shared_rubrics.sh
```

- [ ] **Step 5: Add subprocess-based entrypoint tests**

Create tests modeled after source CPU entrypoint tests, but with target-side expectations:
```python
def test_train_ropd_entrypoint_dryrun_uses_ropd_surface():
    output = run_entrypoint("examples/ropd_trainer/train_ropd.sh")
    assert "reward.reward_manager.name=ropd" in output
    assert "shared_rubrics" not in output
    assert "BLACK_OPD_" not in output
```

- [ ] **Step 6: Verify the dry-run contract**

Run:
```bash
ROPD_DRYRUN=true ROPD_SKIP_REPO_DOTENV=true bash examples/ropd_trainer/train_ropd.sh
uv run --no-sync pytest tests/easyopd/ropd/test_entrypoints.py -v
```
Expected:
```text
Dry-run exits 0 and entrypoint tests pass
```

- [ ] **Step 7: Commit the launch surface**

```bash
git add examples/ropd_trainer tests/easyopd/ropd/test_entrypoints.py
git commit -m "feat: add ropd trainer entrypoint"
```

## Task 7: Add Core CPU Contract Tests For The Migrated Method

**Files:**
- Create: `tests/easyopd/ropd/test_pipeline.py`
- Create: `tests/easyopd/ropd/test_reward_manager.py`
- Create: `tests/easyopd/ropd/test_clients.py`
- Create: `tests/easyopd/ropd/test_judge_config.py`
- Create: `tests/easyopd/ropd/test_judge_provider.py`
- Create: `tests/easyopd/ropd/test_no_legacy_names.py`

- [ ] **Step 1: Port the source CPU tests that validate pure logic, not historical alias behavior**

Good candidates:
```text
test_pipeline_on_cpu.py
test_judge_config_on_cpu.py
test_judge_provider_on_cpu.py
test_judge_provider_resolver_on_cpu.py
test_judge_runtime_builder_on_cpu.py
test_judge_teacher_client_on_cpu.py
```

Do not port alias-preservation tests such as:
```text
test_shared_rubrics_imports_on_cpu.py
test_ropd_env_alias_on_cpu.py
```

- [ ] **Step 2: Rename assertions to the new target-side surface**

Examples:
```python
assert "ropd" in ...
assert "shared_rubrics" not in ...
assert "black_opd" not in ...
```

- [ ] **Step 3: Add one explicit no-legacy-names scan test**

Create a test that reads the new target paths and asserts the public surface is clean:
```python
from pathlib import Path


def test_ropd_public_surface_has_no_legacy_names():
    roots = [
        Path("easyopd/methods/ropd"),
        Path("easyopd/config/ropd"),
        Path("examples/ropd_trainer"),
        Path("docs/algo/ropd.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for root in roots for path in ([root] if root.is_file() else root.rglob("*")) if path.is_file())
    assert "BLACK_OPD_" not in text
    assert "shared_rubrics" not in text
    assert "shared-rubrics" not in text
```

- [ ] **Step 4: Add one reward-manager behavior test that exercises `reward_extra_info`**

Goal:
```python
result = reward_fn(batch, return_dict=True)
assert "reward_tensor" in result
assert "reward_extra_info" in result
```

Keep this CPU-safe by using static or fake clients instead of live providers.

- [ ] **Step 5: Run the focused ROPD test suite**

Run:
```bash
uv run --no-sync pytest tests/easyopd/ropd -v
```
Expected:
```text
All ROPD CPU contract tests pass
```

- [ ] **Step 6: Commit the test suite**

```bash
git add tests/easyopd/ropd
git commit -m "test: add ropd cpu contract coverage"
```

## Task 8: Write User-Facing ROPD Documentation

**Files:**
- Create: `docs/algo/ropd.md`
- Create: `examples/ropd_trainer/README.md`
- Modify: `docs/index.rst`

- [ ] **Step 1: Add the algorithm doc under `docs/algo/ropd.md`**

Sections to include:
```text
Background
What ROPD adds beyond generic OPD
ROPD method structure in EasyOPD
Teacher / rubricator / verifier roles
Config surface
Canonical dry-run command
What is intentionally not migrated
```

- [ ] **Step 2: Keep the example README operational**

It should contain:
```text
Minimal dry-run command
Training command
Required env vars
Recommended DATA_ROOT layout
Why data artifacts are not committed into the repo
```

- [ ] **Step 3: Add `ropd.md` to the docs toctree**

Modify `docs/index.rst` near other algorithm docs:
```rst
   algo/ropd.md
```

- [ ] **Step 4: Keep the docs consistent with the single-entrypoint decision**

All examples must point to:
```bash
bash examples/ropd_trainer/train_ropd.sh
```

Do not reference `training/ppo/train_ropd.sh` or source-repo paths in the target docs, except in clearly labeled migration notes.

- [ ] **Step 5: Sanity-check documentation references**

Run:
```bash
rg -n "train_ropd.sh|shared_rubrics|BLACK_OPD_|black-opd" docs/algo/ropd.md examples/ropd_trainer/README.md docs/index.rst
```
Expected:
```text
Only intentional historical references remain, and the canonical command points to examples/ropd_trainer/train_ropd.sh
```

- [ ] **Step 6: Commit the docs**

```bash
git add docs/algo/ropd.md docs/index.rst examples/ropd_trainer/README.md
git commit -m "docs: add ropd method documentation"
```

## Task 9: Final Verification And Migration Boundary Audit

**Files:**
- Verify: `easyopd/methods/ropd/**`
- Verify: `easyopd/config/ropd/**`
- Verify: `examples/ropd_trainer/**`
- Verify: `tests/easyopd/ropd/**`
- Verify: `docs/algo/ropd.md`
- Verify: `pyproject.toml`
- Verify: `setup.py`
- Verify: `verl/workers/reward_manager/__init__.py`
- Verify: `verl/trainer/ppo/reward.py`

- [ ] **Step 1: Run the focused verification bundle**

Run:
```bash
python -m compileall easyopd/methods/ropd
uv run --no-sync pytest tests/easyopd/ropd tests/workers/reward_manager/test_registry_on_cpu.py -v
ROPD_DRYRUN=true ROPD_SKIP_REPO_DOTENV=true bash examples/ropd_trainer/train_ropd.sh
```
Expected:
```text
compileall succeeds
focused pytest passes
dry-run exits 0 and prints the final command
```

- [ ] **Step 2: Run a legacy-name residue scan across the target surface**

Run:
```bash
rg -n "shared_rubrics|shared-rubrics|BLACK_OPD_|black_opd|training/ppo/train_ropd.sh" easyopd/methods/ropd easyopd/config/ropd examples/ropd_trainer tests/easyopd/ropd docs/algo/ropd.md
```
Expected:
```text
No matches, or only tightly scoped historical notes in docs comments
```

- [ ] **Step 3: Run a prompt-resource packaging audit**

Run:
```bash
rg -n "methods/ropd/prompts/\\*\\.txt" pyproject.toml setup.py
```
Expected:
```text
Both packaging files include the ropd prompt package-data entry
```

- [ ] **Step 4: Audit git diff for unexpected migrated artifacts**

Run:
```bash
git diff --stat
git status --short
```
Expected:
```text
Only planned code, config, docs, script, and test files are modified
No datasets, outputs, caches, checkpoints, or wandb artifacts are present
```

- [ ] **Step 5: Optional wider smoke if dependencies are available**

Run only if the environment is already prepared:
```bash
uv run --no-sync pytest tests/easyopd/simple tests/easyopd/simct tests/easyopd/ropd -v
```
Expected:
```text
Existing EasyOPD method suites still pass alongside ropd
```

- [ ] **Step 6: Prepare the final human review checklist**

The submitter must verify:
```text
- no duplicate upstream PR is being opened
- all changed lines are understood end-to-end
- exact test commands and results are recorded
- AI assistance is disclosed if this work becomes a PR
```

- [ ] **Step 7: Commit the final integrated migration**

```bash
git add easyopd examples docs tests pyproject.toml setup.py verl
git commit -m "feat: migrate ropd into easyopd"
```

## Self-Review

### Spec coverage

- Method package migration is covered by Tasks 2 and 7.
- Prompt/resource migration is covered by Task 4.
- Config migration is covered by Task 5.
- Single canonical entrypoint is covered by Task 6.
- Docs are covered by Task 8.
- CPU tests and dry-run acceptance are covered by Tasks 7 and 9.
- Non-goals are enforced through Task 1 inventory and Task 9 boundary audit.
- The missing EasyOPD-vs-black-opd reward-manager integration seam is explicitly covered by Task 3.

### Placeholder scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- One execution-time branch exists in Task 5 for `sft.yaml`; it includes an explicit fallback action and is not a placeholder.
- One integration-path decision exists in Task 3; the recommended option and acceptance criteria are spelled out before implementation.

### Type and naming consistency

- Target-side public method name is consistently `ropd`.
- Reward-manager registry target is consistently `reward.reward_manager.name=ropd`.
- Package resource path is consistently `easyopd.methods.ropd.prompts`.
- Test root is consistently `tests/easyopd/ropd`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-ropd-migration-to-easyopd-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
