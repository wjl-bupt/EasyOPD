#!/usr/bin/env bash
# Quick launcher for GRPO training - runs in foreground with output to file
set -euo pipefail

export PYTHONPATH="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export RAY_ADDRESS=auto

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
EXPERIMENT_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/benchmark"
DATA_DIR="${EXPERIMENT_DIR}/data"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints/grpo"
STUDENT_MODEL="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-1.5B-Instruct"
REWARD_FN="${EXPERIMENT_DIR}/reward_fn.py"

mkdir -p "${CHECKPOINT_DIR}"

echo "[$(date)] Starting GRPO training..."

exec ${PYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=['${DATA_DIR}/train.parquet']" \
    "data.val_files=['${DATA_DIR}/val.parquet']" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_model_len=1537 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    'trainer.logger=["console"]' \
    trainer.project_name=easyopd_benchmark \
    trainer.experiment_name=grpo \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=200 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.default_local_dir="${CHECKPOINT_DIR}"
