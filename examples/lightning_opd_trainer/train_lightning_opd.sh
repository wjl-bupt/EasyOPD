#!/usr/bin/env bash
# lightning_opd: Offline On-Policy Distillation Training Script
# Paper: https://arxiv.org/abs/2604.13010
#
# Usage:
#   bash examples/lightning_opd_trainer/train_lightning_opd.sh
#
# Dry-run (print command without executing):
#   LIGHTNING_OPD_DRYRUN=true bash examples/lightning_opd_trainer/train_lightning_opd.sh
#
# Environment variables:
#   LIGHTNING_OPD_PROJECT_ROOT  - Override project root (default: auto-detect)
#   LIGHTNING_OPD_DRYRUN        - If "true", print command without executing
#   LIGHTNING_OPD_SKIP_REPO_DOTENV - If "true", skip loading .env (for CI)
#   LIGHTNING_OPD_SFT_CHECKPOINT   - Path to SFT checkpoint (student model)
#   LIGHTNING_OPD_DATA              - Path to precomputed lightning_opd parquet
#   LIGHTNING_OPD_TEACHER_MODEL     - Teacher model path (for consistency check)
#   MODEL_SCALE                     - Model scale: 4b, 8b (default: 4b)

set -xeuo pipefail

# ---- resolve project root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LIGHTNING_OPD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PROJECT_ROOT

# ---- load .env unless skipped ----
if [ "${LIGHTNING_OPD_SKIP_REPO_DOTENV:-}" != "true" ] && [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# ---- model scale ----
MODEL_SCALE="${MODEL_SCALE:-4b}"

# ---- user-adjustable paths ----
SFT_CHECKPOINT="${LIGHTNING_OPD_SFT_CHECKPOINT:-<SET_LIGHTNING_OPD_SFT_CHECKPOINT>}"
TRAIN_DATA="${LIGHTNING_OPD_DATA:-<SET_LIGHTNING_OPD_DATA>}"
TEACHER_MODEL="${LIGHTNING_OPD_TEACHER_MODEL:-}"

# ---- output configuration ----
SAVE_DIR="${LIGHTNING_OPD_SAVE_DIR:-./checkpoint/lightning_opd_${MODEL_SCALE}}"
PROJECT_NAME="${LIGHTNING_OPD_PROJECT_NAME:-lightning_opd}"
EXPERIMENT_NAME="${LIGHTNING_OPD_EXPERIMENT_NAME:-lightning_opd_${MODEL_SCALE}}"

# ---- model-scale defaults ----
case "$MODEL_SCALE" in
    4b)
        ROLLOUT_TP=2
        PPO_MAX_TOKEN_LEN=32768
        ;;
    8b)
        ROLLOUT_TP=4
        PPO_MAX_TOKEN_LEN=32768
        ;;
    *)
        echo "Unknown MODEL_SCALE=$MODEL_SCALE; expected 4b or 8b" >&2
        exit 1
        ;;
esac

# ---- teacher consistency check ----
SFT_TEACHER_MODEL="${LIGHTNING_OPD_SFT_TEACHER_MODEL:-$SFT_CHECKPOINT}"
OPD_TEACHER_MODEL="${LIGHTNING_OPD_OPD_TEACHER_MODEL:-$TEACHER_MODEL}"
CONSISTENCY_STATUS="skipped"
if [ -n "$SFT_TEACHER_MODEL" ] && [ -n "$OPD_TEACHER_MODEL" ]; then
    if python3 - "$SFT_TEACHER_MODEL" "$OPD_TEACHER_MODEL" <<'PY'
import sys
from easyopd.methods.lightning_opd.teacher_consistency import check_teacher_consistency

check_teacher_consistency(sys.argv[1], sys.argv[2])
print("CONSISTENT")
PY
    then
        CONSISTENCY_STATUS="OK (same model)"
    else
        echo "Teacher consistency check failed for training entrypoint." >&2
        exit 1
    fi
fi

# ---- dry-run output ----
if [ "${LIGHTNING_OPD_DRYRUN:-}" = "true" ]; then
    echo "=== lightning_opd Dry-Run ==="
    echo "PROJECT_ROOT:        $PROJECT_ROOT"
    echo "MODEL_SCALE:         $MODEL_SCALE"
    echo "SFT_CHECKPOINT:      $SFT_CHECKPOINT"
    echo "LIGHTNING_OPD_DATA:  $TRAIN_DATA"
    echo "SFT_TEACHER_MODEL:   ${SFT_TEACHER_MODEL:-<not set>}"
    echo "OPD_TEACHER_MODEL:   ${OPD_TEACHER_MODEL:-<not set>}"
    echo "Teacher consistency: $CONSISTENCY_STATUS"
    echo "ROLLOUT_TP:          $ROLLOUT_TP"
    echo ""
    echo "python3 -m verl.trainer.main_ppo \\"
    echo "    --config-path $PROJECT_ROOT/easyopd/config/lightning_opd \\"
    echo "    --config-name base \\"
    echo "    algorithm.adv_estimator=on_policy_distillation \\"
    echo "    data.train_files=['$TRAIN_DATA'] \\"
    echo "    data.train_batch_size=256 \\"
    echo "    data.max_prompt_length=1024 \\"
    echo "    data.max_response_length=4096 \\"
    echo "    data.filter_overlong_prompts=True \\"
    echo "    data.truncation=error \\"
    echo "    actor_rollout_ref.model.path=$SFT_CHECKPOINT \\"
    echo "    actor_rollout_ref.model.use_remove_padding=True \\"
    echo "    actor_rollout_ref.model.enable_gradient_checkpointing=True \\"
    echo "    actor_rollout_ref.actor.optim.lr=2e-6 \\"
    echo "    actor_rollout_ref.actor.ppo_mini_batch_size=256 \\"
    echo "    actor_rollout_ref.actor.use_dynamic_bsz=True \\"
    echo "    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \\"
    echo "    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \\"
    echo "    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \\"
    echo "    actor_rollout_ref.rollout.n=1 \\"
    echo "    trainer.project_name=$PROJECT_NAME \\"
    echo "    trainer.experiment_name=$EXPERIMENT_NAME \\"
    echo "    trainer.n_gpus_per_node=8 \\"
    echo "    trainer.nnodes=1 \\"
    echo "    trainer.total_epochs=1"
    exit 0
fi

# ---- validate required paths ----
if [[ "$SFT_CHECKPOINT" == *"<"* ]] || [[ "$TRAIN_DATA" == *"<"* ]]; then
    echo "Error: Set LIGHTNING_OPD_SFT_CHECKPOINT and LIGHTNING_OPD_DATA" >&2
    exit 1
fi

# ---- launch ----
python3 -m verl.trainer.main_ppo \
    --config-path "$PROJECT_ROOT/easyopd/config/lightning_opd" \
    --config-name base \
    algorithm.adv_estimator=on_policy_distillation \
    "data.train_files=['$TRAIN_DATA']" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$SFT_CHECKPOINT" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=1 \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    "$@"
