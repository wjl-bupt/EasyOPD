#!/bin/bash
# SOD: Step-wise On-policy Distillation Training Script
# Paper: https://arxiv.org/abs/2605.07725
#
# Hardware Requirements: 8x NVIDIA H20 96GB GPUs (>=88GB per GPU)
# Environment: Python 3.10+, vLLM 0.8.x, PyTorch 2.4+
#
# Usage:
#   1. Edit the paths below to match your environment
#   2. Edit examples/sod/sandbox_fusion_tool_config.yaml to set sandbox URL
#   3. Run: bash examples/sod/run_sod.sh
#
# Before running, ensure:
#   - All 8 GPUs are free (nvidia-smi shows 0 MiB usage)
#   - VLLM_USE_V1=1 is set (required for async vLLM rollout)
#   - Sandbox URL is configured in sandbox_fusion_tool_config.yaml

set -e

# ============ Local secrets / paths (optional) ============
# Copy .env.example to .env at the repo root and fill in your real values
# (sandbox URL, model/data paths). .env is git-ignored and never published.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_REPO_ROOT}/.env" ] && set -a && . "${_REPO_ROOT}/.env" && set +a

# ============ Environment Setup ============
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1  # Required: enables vLLM V1 engine for async rollout
# Read from .env / environment; falls back to a placeholder you must replace.
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-https://YOUR_SANDBOX_FUSION_ENDPOINT/run_code}"

# ============ Model Paths (MUST EDIT) ============
# Student model: HuggingFace format SFT checkpoint to be trained
STUDENT_MODEL_PATH="/path/to/workspace/checkpoint/qwen3_1p7b_sft/global_step_115/huggingface"
# Teacher model: HuggingFace format teacher model for KL regularization
TEACHER_MODEL_PATH="/path/to/workspace/download_models/models--Gen-Verse--DemyAgent-4B/snapshots/6a097c80a5b60a106db46d9f72624988b078ad01"

# ============ Dataset Paths (MUST EDIT) ============
TRAIN_DATA="/path/to/workspace/dataset/Gen-Verse/Open-AgentRL-30K/Open-AgentRL-30K.parquet"
VAL_DATA_1="/path/to/workspace/dataset/Gen-Verse/Open-AgentRL-Eval/aime2025/aime_2025_problems.parquet"
VAL_DATA_2="/path/to/workspace/dataset/Gen-Verse/Open-AgentRL-Eval/aime2024/aime_2024_problems.parquet"

# ============ Output Configuration ============
SAVE_DIR="/path/to/workspace/SOD_Merge_Framework/checkpoint/sod_easyopd_clone"
PROJECT_NAME="sod_easyopd_clone"
EXPERIMENT_NAME="sod_easyopd_clone"

# ============ Training ============
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.002 \
    +algorithm.filter_groups.enable=False \
    +algorithm.filter_groups.metric=seq_reward \
    +algorithm.filter_groups.reward_std_threshold=0.0 \
    +algorithm.token_kl_reg.enable=True \
    +algorithm.token_kl_reg.gamma=1.0 \
    +algorithm.token_kl_reg.beta_min=0.0 \
    +algorithm.token_kl_reg.beta_max=0.10 \
    +algorithm.token_kl_reg.stepwise_enable=True \
    +algorithm.token_kl_reg.stepwise_epsilon=1e-6 \
    +algorithm.token_kl_reg.stepwise_delta=0.2 \
    +algorithm.token_kl_reg.stepwise_opd_coef=1.0 \
    "data.train_files=['${TRAIN_DATA}']" \
    "data.val_files=['${VAL_DATA_1}','${VAL_DATA_2}']" \
    data.return_raw_chat=True \
    data.train_batch_size=64 \
    data.max_prompt_length=2560 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.custom_cls.path=examples/sod/reward.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=examples/sod/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=${STUDENT_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=23040 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.ref.model.path=${TEACHER_MODEL_PATH} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=92160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=16 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/sod/sandbox_fusion_tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=1024 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=20480 \
    "trainer.logger=['console']" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.save_freq=10 \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.test_freq=10 \
    trainer.total_epochs=1
