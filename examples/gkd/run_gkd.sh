#!/bin/bash
set -x
export PYTHONUNBUFFERED=1

# =============================================================================
# GKD: Generalized Knowledge Distillation (On-Policy Distillation)
# Paper: https://arxiv.org/abs/2306.13649 (ICLR 2024)
#
# This script demonstrates GKD training for knowledge distillation:
# - On-policy: student generates sequences, teacher provides dense feedback
# - Loss: Generalized JSD (beta interpolates forward/reverse KL)
# - Supports both policy-gradient and supervised distillation modes
# =============================================================================

# ============ Parse arguments ============
MODEL_PATH="Qwen/Qwen2-0.5B-Instruct"
TEACHER_MODEL_PATH="Qwen/Qwen2-1.5B-Instruct"
BETA=0.5
TEMPERATURE=1.0
ACTOR_LR=5e-6
USE_POLICY_GRADIENT=True
USE_TASK_REWARDS=True
DISTILLATION_LOSS_COEF=1.0
LOSS_MODE="gkd"
MAX_RESPONSE_LENGTH=512
MAX_PROMPT_LENGTH=1024
NNODES=1
TEACHER_TP=2
EXP_NAME="gkd-distill"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --teacher_model)
            TEACHER_MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --loss_mode)
            LOSS_MODE="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --teacher_tp)
            TEACHER_TP="$2"
            shift 2
            ;;
        --actor_lr)
            ACTOR_LR="$2"
            shift 2
            ;;
        --use_policy_gradient)
            USE_POLICY_GRADIENT="$2"
            shift 2
            ;;
        --use_task_rewards)
            USE_TASK_REWARDS="$2"
            shift 2
            ;;
        --distillation_loss_coef)
            DISTILLATION_LOSS_COEF="$2"
            shift 2
            ;;
        --max_response_length)
            MAX_RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --max_prompt_length)
            MAX_PROMPT_LENGTH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# ============ Environment ============
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Running GKD: $EXP_NAME"
echo "Student Model: $MODEL_PATH"
echo "Teacher Model: $TEACHER_MODEL_PATH"
echo "Beta (JSD interpolation): $BETA"
echo "Temperature: $TEMPERATURE"
echo "Loss Mode: $LOSS_MODE"
echo "Use Policy Gradient: $USE_POLICY_GRADIENT"

# ============ Compute teacher resource pool ============
# Teacher uses TEACHER_TP GPUs per replica
TEACHER_POOL_SIZE=$((8 * NNODES))
# Ensure teacher pool is divisible by TP
if [ $((TEACHER_POOL_SIZE % TEACHER_TP)) -ne 0 ]; then
    echo "ERROR: Teacher pool size ($TEACHER_POOL_SIZE) must be divisible by teacher TP ($TEACHER_TP)"
    exit 1
fi

# ============ Launch Training ============
python3 -m verl.trainer.main_ppo \
    data.prompt_key=content \
    data.train_files=/tmp/gkd_train.parquet \
    data.val_files=/tmp/gkd_test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=1 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    distillation.enabled=True \
    distillation.n_gpus_per_node=8 \
    distillation.nnodes=${NNODES} \
    distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL_PATH} \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5 \
    distillation.distillation_loss.loss_mode=${LOSS_MODE} \
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT} \
    distillation.distillation_loss.use_task_rewards=${USE_TASK_REWARDS} \
    distillation.distillation_loss.distillation_loss_coef=${DISTILLATION_LOSS_COEF} \
    distillation.distillation_loss.gkd_beta=${BETA} \
    distillation.distillation_loss.gkd_temperature=${TEMPERATURE} \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=GKD \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=5 \
    trainer.test_freq=10000000000 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    trainer.default_local_dir=/tmp/${EXP_NAME} \
    "${@:1}"
