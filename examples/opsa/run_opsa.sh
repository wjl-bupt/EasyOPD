#!/usr/bin/env bash
# OPSA: On-Policy Self-Distillation for Safety Alignment
# Paper: https://arxiv.org/abs/2605.15239
#
# This script demonstrates OPSA training for safety alignment:
# - Self-distillation: teacher = frozen copy of student (no separate teacher model)
# - Refusal-decision window: concentrate gradient on early safety-critical tokens
# - Type-conditional privileged contexts: harmful vs. benign prompts
#
# Usage:
#   1. Install EasyOPD: pip install -e .
#   2. Prepare safety data: python examples/opsa/prepare_safety_data.py
#   3. Edit MODEL_PATH and DATA_PATH below (or override via environment)
#   4. Run: bash examples/opsa/run_opsa.sh
#
# Config reference: easyopd/config/opsa.yaml

set -euo pipefail

# ============ Environment ============
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ============ Configuration ============
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
DATA_PATH="${DATA_PATH:-data/opsa/safechain_train.parquet}"
# No separate validation split (matches the original OPSA setup); reuse the
# training file so verl trainer's required val_files is satisfied.
VAL_DATA_PATH="${VAL_DATA_PATH:-${DATA_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/opsa}"

# GPU Configuration
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"

# Training Hyperparameters
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
ACTOR_LR="${ACTOR_LR:-1e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"

# OPSA-Specific Parameters (aligned with original OPSA distillation_safety.yaml)
OPSA_TEMPERATURE="${OPSA_TEMPERATURE:-1.0}"
OPSA_WINDOW_SIZE="${OPSA_WINDOW_SIZE:-32}"
OPSA_DECAY_TYPE="${OPSA_DECAY_TYPE:-linear}"
OPSA_MIN_WEIGHT="${OPSA_MIN_WEIGHT:-0.1}"
OPSA_LOSS_COEF="${OPSA_LOSS_COEF:-1.0}"
OPSA_KL_TYPE="${OPSA_KL_TYPE:-mixed}"
OPSA_MIXED_KL_WEIGHT="${OPSA_MIXED_KL_WEIGHT:-0.5}"
OPSA_TOPK_LOGITS_K="${OPSA_TOPK_LOGITS_K:-512}"

# Optimizer / scheduler (original OPSA: lr=1e-5, weight_decay=0.01, max_grad_norm=1.0,
# LinearLR warmup over 10 steps from 0.1 -> 1.0, then ConstantLR)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"

# Rollout Configuration (original OPSA: max_total_sequence_length=4096, temperature=0.6)
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

echo "============================================"
echo "OPSA: On-Policy Self-Distillation for Safety Alignment"
echo "============================================"
echo "Model:  ${MODEL_PATH}"
echo "Data:   ${DATA_PATH}"
echo "Val:    ${VAL_DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs:   ${GPUS_PER_NODE} x ${NNODES} nodes"
echo "OPSA:   window_size=${OPSA_WINDOW_SIZE}, temperature=${OPSA_TEMPERATURE}, decay=${OPSA_DECAY_TYPE}"
echo "============================================"

python3 -m verl.trainer.main_ppo \
    data.train_files="['${DATA_PATH}']" \
    data.val_files="['${VAL_DATA_PATH}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation=right \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    +actor_rollout_ref.actor.opsa_enable=true \
    +actor_rollout_ref.actor.opsa_temperature=${OPSA_TEMPERATURE} \
    +actor_rollout_ref.actor.opsa_window_size=${OPSA_WINDOW_SIZE} \
    +actor_rollout_ref.actor.opsa_decay_type=${OPSA_DECAY_TYPE} \
    +actor_rollout_ref.actor.opsa_min_weight=${OPSA_MIN_WEIGHT} \
    +actor_rollout_ref.actor.opsa_use_window_weighting=true \
    +actor_rollout_ref.actor.opsa_distillation_loss_coef=${OPSA_LOSS_COEF} \
    +actor_rollout_ref.actor.opsa_loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.opsa_kl_type=${OPSA_KL_TYPE} \
    +actor_rollout_ref.actor.opsa_mixed_kl_weight=${OPSA_MIXED_KL_WEIGHT} \
    +actor_rollout_ref.actor.opsa_topk_logits_k=${OPSA_TOPK_LOGITS_K} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.project_name=opsa \
    trainer.experiment_name=opsa_${MODEL_PATH##*/}_window${OPSA_WINDOW_SIZE} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    "trainer.logger=['console']" \
    "${@:1}"
