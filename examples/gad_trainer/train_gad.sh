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
