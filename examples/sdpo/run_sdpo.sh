#!/bin/bash
set -x
export PYTHONUNBUFFERED=1

# =============================================================================
# SDPO: Self-Distilled Policy Optimization
# Paper: https://arxiv.org/abs/2601.20802 (Hübotter et al., 2026)
#
# This script demonstrates SDPO training:
# - Self-distillation from the model's own high-reward trajectories
# - No external teacher model required (uses EMA of student as self-teacher)
# - Supports reprompting with successful demonstrations and environment feedback
# - Loss: Generalized JSD between student and self-teacher
# =============================================================================

# ============ Parse arguments ============
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
ALPHA=0.5
ACTOR_LR=1e-5
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
DISTILLATION_TOPK=100
IS_CLIP=2.0
TEACHER_UPDATE_RATE=0.05
SUCCESS_REWARD_THRESHOLD=1.0
DONT_REPROMPT_ON_SELF_SUCCESS=True
REMOVE_THINKING=True
INCLUDE_FEEDBACK=True
MAX_REPROMPT_LEN=10240
MAX_RESPONSE_LENGTH=8192
MAX_PROMPT_LENGTH=2048
NNODES=1
EXP_NAME="sdpo-self-distill"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --actor_lr)
            ACTOR_LR="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --rollout_n)
            ROLLOUT_N="$2"
            shift 2
            ;;
        --distillation_topk)
            DISTILLATION_TOPK="$2"
            shift 2
            ;;
        --is_clip)
            IS_CLIP="$2"
            shift 2
            ;;
        --teacher_update_rate)
            TEACHER_UPDATE_RATE="$2"
            shift 2
            ;;
        --success_threshold)
            SUCCESS_REWARD_THRESHOLD="$2"
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
        --nnodes)
            NNODES="$2"
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

echo "Running SDPO: $EXP_NAME"
echo "Model: $MODEL_PATH"
echo "Alpha (JSD interpolation): $ALPHA"
echo "Distillation Top-k: $DISTILLATION_TOPK"
echo "IS Clip: $IS_CLIP"
echo "Teacher Update Rate: $TEACHER_UPDATE_RATE"
echo "Rollout N: $ROLLOUT_N"

# ============ Launch Training ============
python3 -m verl.trainer.main_ppo \
    data.prompt_key=content \
    data.train_files=/tmp/sdpo_train.parquet \
    data.val_files=/tmp/sdpo_test.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=1 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
    actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK} \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
    actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=${SUCCESS_REWARD_THRESHOLD} \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
    actor_rollout_ref.actor.self_distillation.is_clip=${IS_CLIP} \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LEN} \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=${REMOVE_THINKING} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=${INCLUDE_FEEDBACK} \
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=SDPO \
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
