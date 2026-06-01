# ROPD: Rubric-based On-policy Distillation

## Background

ROPD (Rubric-based On-policy Distillation) is the maintained mainline of the
research code previously hosted at `black-opd`. It plugs into verl's standard
`verl.trainer.main_ppo` entrypoint through a custom reward manager that
evaluates each rollout against a teacher-grounded, dynamically-constructed
rubric instead of a fixed scalar score.

## What ROPD adds beyond generic on-policy distillation

Generic on-policy distillation (OPD) trains the student to imitate the teacher's
logits on rollouts the student itself generated. ROPD keeps that "on-policy"
property and adds a **rubric-shaped reward**:

1. A **teacher** model (or a pre-computed teacher index) emits a reference
   answer for the prompt.
2. A **rubricator** model generates a small set of binary criteria that
   distinguish the teacher's answer from the student's.
3. A **verifier** model applies that rubric to both the teacher answer and
   the student rollout, producing two final scores.
4. The reward for the student rollout is the normalized **score gap**
   `(student_score - teacher_score) / rubric.maximum_score`.

This makes the reward signal grounded in concrete, per-prompt criteria rather
than relying on a single scalar teacher rating, which empirically yields
denser and more stable RL signals.

## ROPD method structure in EasyOPD

```
easyopd/methods/ropd/
  __init__.py              # ROPDMethod metadata + register() hook
  pipeline.py              # ROPDGroup, ROPDRollout, ROPDPipeline
  prompt_utils.py          # raw-prompt normalization + template rendering
  prompts/                 # packaged rubricator / verifier templates (.txt)
  prompts/__init__.py      # build_rubricator_prompt / build_verifier_prompt
  clients.py               # ROPDClientConfig + judge client builders
  teacher_index.py         # OfflineTeacherIndex + fingerprint helpers
  reward_manager.py        # ROPDRewardManager (registered as `ropd`)
  judge/                   # provider resolution, transport, schema, scheduler
  utils/eval_package.py    # diagnostic archiving helpers
```

The reward manager is registered under the name `ropd` into verl's
`verl.workers.reward_manager` registry, so the standard launcher resolves it via
`reward_model.reward_manager=ropd`.

## Teacher / rubricator / verifier roles

Each of the three judge roles can be configured independently. Provider
profiles, base URLs, and API key env vars are declared in
`easyopd/config/ropd/judge_providers.yaml`; per-role models and timeouts live in
the same file under the `roles:` block. ROPD-specific knobs (max group
concurrency, quality-gate thresholds, scheduler) live in
`easyopd/config/ropd/judge.yaml`.

| Role | Purpose | Provider options |
|---|---|---|
| `teacher` | Produces the reference answer per prompt. | `offline_index` (pre-computed JSONL), `openai_compatible`, `static` (debug) |
| `rubricator` | Builds the binary rubric per prompt. | `openai_compatible`, `static` (debug) |
| `verifier` | Scores teacher and student answers under the rubric. | `openai_compatible`, `static` (debug) |

`offline_index` requires a fingerprinted JSONL produced by the source ROPD
teacher-index pipeline. The fingerprint pins provider/model/temperature so a
stale teacher index cannot silently mismatch the current configuration.

## Config surface

Two repo-side YAML templates are provided:

- `easyopd/config/ropd/base.yaml` — ROPD-specific overrides on top of the
  default verl PPO trainer config. Sets `reward_model.reward_manager=ropd` and
  the student model defaults.
- `easyopd/config/ropd/judge.yaml` — judge knobs that live under
  `reward_model.reward_kwargs.ropd.*`, including the provider resolution spec
  path and quality-gate parameters.
- `easyopd/config/ropd/judge_providers.yaml` — provider/profile/role resolution
  spec consumed by `JudgeProviderResolver`.

An optional `easyopd/config/ropd/sft.yaml` template is included for a future
offline-policy SFT warmup; it is not wired to a launcher in this phase of the
migration.

## Canonical dry-run command

To preview the assembled command without running it:

```bash
ROPD_DRYRUN=true ROPD_SKIP_REPO_DOTENV=true bash examples/ropd_trainer/train_ropd.sh
```

The dry-run prints the detected `PROJECT_ROOT`, config template paths, judge
provider spec path, the data files it would load, the student model source,
and the final assembled Python command.

## Required environment

| Variable | Purpose |
|---|---|
| `DATA_ROOT` | Base directory containing the training data parquet shards. |
| `ROPD_TRAIN_TASK` | Task subdirectory under `DATA_ROOT`. |
| `ROPD_STUDENT_MODEL` | Student model path / HF repo id (defaults to `Qwen/Qwen2.5-7B-Instruct`). |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | Backing the `openai_default` profile. |
| `ROPD_VLLM_API_KEY` / `ROPD_VLLM_BASE_URL` | Backing the `local_vllm` profile. |

The launcher honors the standard verl Hydra override mechanism. Pass extra
overrides through `ROPD_EXTRA_OVERRIDES="key1=v1 key2=v2"`.

## What is intentionally not migrated

The first-phase migration deliberately excludes:

- Source-repo `outputs/`, `logs/`, `wandb/`, `.cache/`, `.pytest_cache/`
  artifacts.
- The full `datasets/unified/*` data tree (only the loading paths are
  templated; the data itself stays out of repo).
- ROPD ablation directories that are not part of the maintained mainline.
- Source-repo legacy alias scripts and environment-variable aliases retained
  for historical compatibility.
- The verifier replay / evaluation tooling beyond what the reward path itself
  needs at training time.

If you need any of those, work from the original `black-opd` checkout — they
were intentionally left out of EasyOPD's surface to keep this method module
small and reviewable.
