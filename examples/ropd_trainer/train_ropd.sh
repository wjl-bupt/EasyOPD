#!/usr/bin/env bash
# EasyOPD `ropd` — Rubric-based On-policy Distillation launch script.
#
# Mirrors verl's standard `verl.trainer.main_ppo` entrypoint and wires the
# `ropd` reward manager registered by `easyopd.methods.ropd`.
#
# Required toggles handled by this script:
#   - reward_model.reward_manager=ropd
#   - +reward_model.reward_kwargs.ropd.provider_resolution.spec_path=easyopd/config/ropd/judge_providers.yaml
#
# Environment variables (all optional, all prefixed `ROPD_*`):
#   ROPD_DRYRUN             "true" to print the assembled command without running.
#   ROPD_SKIP_REPO_DOTENV   "true" to skip loading the repo `.env` (CI / tests).
#   ROPD_PROJECT_ROOT       Override the detected project root.
#   ROPD_CONFIG             ROPD config template name (default: base).
#   ROPD_JUDGE_CONFIG       Judge knobs config template name (default: judge).
#   ROPD_JUDGE_PROVIDERS    Path to the judge provider spec YAML.
#   ROPD_STUDENT_MODEL      Override student model path.
#   ROPD_TRAIN_TASK         Subdirectory under DATA_ROOT for the active task.
#   DATA_ROOT               Base directory containing training data.
#   ROPD_EXTRA_OVERRIDES    Additional Hydra overrides space-separated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ROPD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

is_true() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

# ----- optional repo .env -----
if ! is_true "${ROPD_SKIP_REPO_DOTENV:-false}" && [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ----- resolve configuration ------
ROPD_CONFIG_NAME="${ROPD_CONFIG:-base}"
ROPD_JUDGE_CONFIG_NAME="${ROPD_JUDGE_CONFIG:-judge}"
ROPD_JUDGE_PROVIDERS_PATH="${ROPD_JUDGE_PROVIDERS:-easyopd/config/ropd/judge_providers.yaml}"
ROPD_STUDENT_MODEL_PATH="${ROPD_STUDENT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA_ROOT="${DATA_ROOT:-datasets/unified}"
ROPD_TRAIN_TASK_NAME="${ROPD_TRAIN_TASK:-math/dapo-math-17k}"
ROPD_EXTRA_OVERRIDES_VALUE="${ROPD_EXTRA_OVERRIDES:-}"

DATA_TRAIN_FILES="$DATA_ROOT/$ROPD_TRAIN_TASK_NAME/train.parquet"
DATA_VAL_FILES="$DATA_ROOT/$ROPD_TRAIN_TASK_NAME/val.parquet"

JUDGE_PROVIDER_SOURCE="$PROJECT_ROOT/$ROPD_JUDGE_PROVIDERS_PATH"
if [[ ! -f "$JUDGE_PROVIDER_SOURCE" ]]; then
    JUDGE_PROVIDER_SOURCE="$ROPD_JUDGE_PROVIDERS_PATH"
fi

# ----- assemble Hydra command -----
HYDRA_OVERRIDES=(
    "reward_model.reward_manager=ropd"
    "+reward_model.reward_kwargs.ropd.provider_resolution.spec_path=$ROPD_JUDGE_PROVIDERS_PATH"
    "+reward_model.reward_kwargs.ropd.provider_resolution.entrypoint=train"
    "data.train_files=$DATA_TRAIN_FILES"
    "data.val_files=$DATA_VAL_FILES"
    "actor_rollout_ref.model.path=$ROPD_STUDENT_MODEL_PATH"
)
if [[ -n "$ROPD_EXTRA_OVERRIDES_VALUE" ]]; then
    read -r -a EXTRA_TOKENS <<< "$ROPD_EXTRA_OVERRIDES_VALUE"
    HYDRA_OVERRIDES+=("${EXTRA_TOKENS[@]}")
fi

PYTHON_CMD=(
    python3 -m verl.trainer.main_ppo
    "${HYDRA_OVERRIDES[@]}"
)

# ----- dry-run preview -----
if is_true "${ROPD_DRYRUN:-false}"; then
    echo "PROJECT_ROOT=$PROJECT_ROOT"
    echo "Config template=easyopd/config/ropd/${ROPD_CONFIG_NAME}.yaml"
    echo "Judge knobs template=easyopd/config/ropd/${ROPD_JUDGE_CONFIG_NAME}.yaml"
    echo "Judge provider spec=$JUDGE_PROVIDER_SOURCE"
    echo "DATA_ROOT=$DATA_ROOT"
    echo "Student model source=$ROPD_STUDENT_MODEL_PATH"
    echo "Reward manager=ropd"
    echo "Final python command:"
    printf '  %s' "${PYTHON_CMD[@]}"
    printf '\n'
    exit 0
fi

cd "$PROJECT_ROOT"
exec "${PYTHON_CMD[@]}"
