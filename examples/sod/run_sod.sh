#!/bin/bash
set -x

# ============================================================
# SOD: Step-wise On-policy Distillation Training Script
# EasyOPD Integration
#
# This script trains a small language model agent using SOD,
# which combines GRPO with step-wise weighted OPD.
#
# Paper: https://arxiv.org/abs/2605.07725
# ============================================================

# [Optional] Uncomment and set your WandB API key for logging
# export WANDB_API_KEY="<Your_WandB_API_Key>"
# python -c "import wandb; wandb.login()"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# ---- Data Paths ----
open_agent_rl=${OPEN_AGENT_RL:-<Your_RL_Training_Data_Path>}
aime_2024=${AIME_2024:-<Your_AIME2024_Eval_Data_Path>}
aime_2025=${AIME_2025:-<Your_AIME2025_Eval_Data_Path>}

# ---- Model Paths ----
# Student: SFT checkpoint (HuggingFace format)
student_model_path=${STUDENT_MODEL_PATH:-<Your_Student_Model_Path>}
# Teacher: GRPO-optimized checkpoint (HuggingFace format)
teacher_model_path=${TEACHER_MODEL_PATH:-<Your_Teacher_Model_Path>}

# ---- Tool Config ----
tool_config_path=${TOOL_CONFIG_PATH:-recipe/demystify/sandbox_fusion_tool_config.yaml}

# ---- Project Naming ----
project_name=${PROJECT_NAME:-sod}
experiment_name=${EXPERIMENT_NAME:-qwen3_1p7b_sod}

default_local_dir=${DEFAULT_LOCAL_DIR:-./checkpoint/$experiment_name}

train_files="['$open_agent_rl']"
test_files="['$aime_2025', '$aime_2024']"

# ---- GRPO Parameters ----
adv_estimator=grpo
use_kl_in_reward=False
teacher_kl_coef=0.002

use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode=token-mean
reward_manager=dapo
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

enable_filter_groups=False
filter_groups_metric=seq_reward
filter_groups_reward_std_threshold=0.0

# ---- Step-wise Weighted OPD Parameters (SOD Core) ----
# Enable the token_kl_reg module (required for OPD)
token_kl_reg_enable=True
# Enable step-wise mode
stepwise_enable=True
# epsilon: numerical stability for d_k ratio computation (Eq. 7)
stepwise_epsilon=1e-6
# delta: upper bound offset, w_k <= 1 + delta (Eq. 7)
stepwise_delta=0.2
# opd_coef: global coefficient for the step-wise OPD term
stepwise_opd_coef=1.0
# Legacy parameters (not used in stepwise mode, kept for compatibility)
token_kl_gamma=1.0
token_kl_beta_min=0.0
token_kl_beta_max=0.10

# ---- Training Configuration ----
max_turns=16
max_prompt_length=2560
max_response_length=20480
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=16
n_resp_per_prompt_val=32

infer_tp=4
train_sp=4
offload=False

actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 1))
log_prob_max_token_len_per_gpu=$((actor_max_token_len_per_gpu * 4))

# ---- Launch Training ----
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$teacher_kl_coef \
    +algorithm.filter_groups.enable=$enable_filter_groups \
    +algorithm.filter_groups.metric=$filter_groups_metric \
    +algorithm.filter_groups.reward_std_threshold=$filter_groups_reward_std_threshold \
    +algorithm.token_kl_reg.enable=$token_kl_reg_enable \
    +algorithm.token_kl_reg.gamma=$token_kl_gamma \
    +algorithm.token_kl_reg.beta_min=$token_kl_beta_min \
    +algorithm.token_kl_reg.beta_max=$token_kl_beta_max \
    +algorithm.token_kl_reg.stepwise_enable=$stepwise_enable \
    +algorithm.token_kl_reg.stepwise_epsilon=$stepwise_epsilon \
    +algorithm.token_kl_reg.stepwise_delta=$stepwise_delta \
    +algorithm.token_kl_reg.stepwise_opd_coef=$stepwise_opd_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.custom_cls.path=recipe/demystify/reward.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/demystify/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    reward_model.reward_manager=$reward_manager \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=$enable_overlong_buffer \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$overlong_buffer_len \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=$overlong_penalty_factor \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=$max_response_length \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.save_freq=10 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=10 \
    trainer.total_epochs=1 "$@"
