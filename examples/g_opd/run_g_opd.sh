#!/bin/bash
set -x
export PYTHONUNBUFFERED=1

# ============================================================
# G-OPD: Generalized On-Policy Distillation with Reward Extrapolation
# Paper: https://arxiv.org/abs/2602.12125
#
# This script demonstrates ExOPD (lambda=1.25) single-teacher distillation.
# For standard OPD, set lambda_vals=1.0 and remove base_model_path.
# ============================================================

export WANDB_API_KEY=""
export WANDB_MODE=online

# ============ Configure paths ============
STUDENT_MODEL_PATH="Qwen/Qwen3-1.7B"
TEACHER_MODEL_PATH="Qwen/Qwen3-4B-Non-Thinking-RL-Math"
# Base model for reward normalization (typically student's initial state)
BASE_MODEL_PATH="Qwen/Qwen3-1.7B"

TRAIN_DATA="<path_to_training_data>/train.parquet"
VAL_DATA_1="<path_to_eval_data>/AIME2024/test.parquet"
VAL_DATA_2="<path_to_eval_data>/AIME2025/test.parquet"

test_files="['$VAL_DATA_1', '$VAL_DATA_2']"

# ============ ExOPD Training (lambda=1.25) ============
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$STUDENT_MODEL_PATH \
    +actor_rollout_ref.model.base_model_path=$BASE_MODEL_PATH \
    +actor_rollout_ref.ref.model.path=$TEACHER_MODEL_PATH \
    +actor_rollout_ref.ref.model.base_model_path=$BASE_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.lambda_vals=1.25 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.project_name='on-policy-distillation' \
    trainer.experiment_name='g_opd_exopd_lambda1.25' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=./g_opd_checkpoints/ExOPD-Lambda1.25 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@
