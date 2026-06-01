# EasyOPD `ropd` Trainer

Canonical launcher for the **ROPD** (Rubric-based On-policy Distillation) method
inside EasyOPD. The Python implementation lives in
`easyopd/methods/ropd/`; the reward manager is registered into verl's
reward-manager registry under the name `ropd`, so the standard `verl.trainer.main_ppo`
entrypoint can resolve it via `reward_model.reward_manager=ropd`.

## Quick start

Sanity-check the assembled command without running it:

```bash
ROPD_DRYRUN=true ROPD_SKIP_REPO_DOTENV=true bash examples/ropd_trainer/train_ropd.sh
```

Run a real training job (requires data, a teacher index, and any provider keys
needed by `easyopd/config/ropd/judge_providers.yaml`):

```bash
DATA_ROOT=/path/to/datasets \
ROPD_TRAIN_TASK=math/dapo-math-17k \
ROPD_STUDENT_MODEL=Qwen/Qwen2.5-7B-Instruct \
bash examples/ropd_trainer/train_ropd.sh
```

## Environment variables

All ROPD environment variables use the `ROPD_*` prefix. There are no legacy
environment aliases retained from the source repo.

| Variable | Purpose |
|---|---|
| `ROPD_DRYRUN` | Print the assembled command and exit (no Python invocation). |
| `ROPD_SKIP_REPO_DOTENV` | Skip loading `$PROJECT_ROOT/.env`. Used by CI / tests. |
| `ROPD_PROJECT_ROOT` | Override the detected project root. |
| `ROPD_CONFIG` | ROPD base template name (default `base`, resolves to `easyopd/config/ropd/base.yaml`). |
| `ROPD_JUDGE_CONFIG` | Judge knobs template name (default `judge`). |
| `ROPD_JUDGE_PROVIDERS` | Provider spec YAML path (default `easyopd/config/ropd/judge_providers.yaml`). |
| `ROPD_STUDENT_MODEL` | Student model path / HF repo id. |
| `ROPD_TRAIN_TASK` | Subdirectory under `DATA_ROOT` for the active task. |
| `DATA_ROOT` | Base directory containing training data parquet shards. |
| `ROPD_EXTRA_OVERRIDES` | Space-separated extra Hydra overrides (e.g. `actor_rollout_ref.actor.optim.lr=5e-7`). |

## Recommended `DATA_ROOT` layout

The launcher resolves training data as:

```
$DATA_ROOT/$ROPD_TRAIN_TASK/train.parquet
$DATA_ROOT/$ROPD_TRAIN_TASK/val.parquet
$DATA_ROOT/$ROPD_TRAIN_TASK/artifacts/teacher_index/teacher-index.jsonl
```

EasyOPD does not ship any of those data products. Build them with the offline
teacher-index pipeline from the source ROPD reference workflow (see the
ROPD method doc in `docs/algo/ropd.md`).

## What is **not** migrated into this folder

The first-phase migration intentionally excludes:

- Source-repo top-level `outputs/`, `logs/`, `wandb/`, `.cache/` artifacts.
- `datasets/unified/*` and any benchmark output snapshots.
- ROPD ablation directories that are not part of the maintained mainline.
- Legacy alias scripts (`validate_ropd.sh`, `launch_ropd_gpu_sweep_tmux.sh`,
  the old shared-rubric trainer wrapper).
- Source-repo `_paths.sh` / `wandb.env.local` helpers; this script inlines a
  minimal `.env` loader instead.

If you need to reproduce historical ROPD runs that depended on any of the
above, work from the original `black-opd` checkout, not this folder.
