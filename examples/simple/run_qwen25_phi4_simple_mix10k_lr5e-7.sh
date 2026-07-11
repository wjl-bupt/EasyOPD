#!/usr/bin/env bash
# EasyOPD `simple` — Cross-Tokenizer On-Policy Distillation
#
# Mirrors the KDFlow reference script:
#   KDFlow/scripts/ctopd_phi4/qwen25_phi4_simple_mix10k_lr5e-7.sh
#
# Teacher : Qwen2.5-7B-Instruct
# Student : phi-4-mini (SFT warmup checkpoint)
# Algo    : simple (cross-tokenizer KD on overlap sub-vocab, reverse KL)
# Data    : mixed_math_code_10k
# 1 epoch, lr=5e-7, global batch 64
#
# This is the validation experiment for the EasyOPD `simple` migration:
# the loss curve / final eval should be comparable to the KDFlow run.

set -xeuo pipefail

# ============ Paths (sync student checkpoint to local SSD, like KDFlow) ============
LOCAL_STUDENT_DIR=${LOCAL_STUDENT_DIR:-$HOME/workspace/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6}
REMOTE_STUDENT_DIR=${REMOTE_STUDENT_DIR:-/path/to/workspace/workspace/KDFlow/output/ckpts/phi4-mini-sft-warmup-10k-qwen-lr2e-6}
STUDENT_CKPT_TAG=${STUDENT_CKPT_TAG:-checkpoint-40}

if [ ! -d "${LOCAL_STUDENT_DIR}/${STUDENT_CKPT_TAG}" ]; then
    echo "[run_qwen25_phi4_simple] syncing student checkpoint to local SSD..."
    mkdir -p "${LOCAL_STUDENT_DIR}"
    rsync -ah --progress \
        "${REMOTE_STUDENT_DIR}/${STUDENT_CKPT_TAG}/" \
        "${LOCAL_STUDENT_DIR}/${STUDENT_CKPT_TAG}/"
fi

STUDENT_MODEL=${STUDENT_MODEL:-${LOCAL_STUDENT_DIR}/${STUDENT_CKPT_TAG}}
TEACHER_MODEL=${TEACHER_MODEL:-/path/to/models/Qwen2.5-7B-Instruct}

# ============ Cluster ============
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# KDFlow: teacher_dp_size=8, teacher_tp_size=1 (8 GPUs co-located with student).
# Verl's distillation pool is separate from the student pool, so we reserve a
# dedicated slice. Override TEACHER_WORLD_SIZE if you want a different split.
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}

# ============ Distillation knobs (mirror KDFlow) ============
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-simple}
# kd_loss_fn=rkl in KDFlow -> reverse KL on the overlap sub-vocab.
kl_direction=${KL_DIRECTION:-reverse}
# kd_ratio=1.0 in KDFlow -> pure distillation, no task reward / no SFT term.
use_policy_gradient=${USE_POLICY_GRADIENT:-False}

# ============ Train / data ============
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
# KDFlow: max_len=8192 with generate_max_len=4096 -> roughly half prompt, half response.
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
actor_lr=${ACTOR_LR:-5e-7}
total_epochs=${TOTAL_EPOCHS:-1}
lr_warmup_ratio=${LR_WARMUP_RATIO:-0.05}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.6}
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.6}

save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-50}
log_freq=${LOG_FREQ:-5}

project_name=${PROJECT_NAME:-easyopd_simple_xtok}
experiment_name=${EXPERIMENT_NAME:-qwen25_phi4_simple_mix10k_lr5e-7}

# ============ Data files (override if your verl-style parquets live elsewhere) ============
# Verl ingests parquet, not raw HF datasets — convert your mixed_math_code_10k once
# into prompt/response parquet files (see verl docs `examples/data_preprocess/`)
# and point these env vars at them.
TRAIN_PARQUET=${TRAIN_PARQUET:-$HOME/data/mixed_math_code_10k/train.parquet}
VAL_PARQUET=${VAL_PARQUET:-$HOME/data/mixed_math_code_10k/test.parquet}
train_files="['${TRAIN_PARQUET}']"
val_files="['${VAL_PARQUET}']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_ratio}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# `simple` is a cross-tokenizer KD: teacher runs an independent HF forward.
# verl's `validate_and_prepare_for_distillation` skips the topk / response_length=1
# rewrite in this mode (gated by DistillationLossSettings.use_cross_tokenizer).
EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.distillation_loss_coef=1.0
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.cross_tokenizer_kl_direction=${kl_direction}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
