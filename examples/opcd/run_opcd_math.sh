#!/bin/bash
set -x
export PYTHONUNBUFFERED=1

# =============================================================================
# OPCD: On-Policy Context Distillation for Language Models
# Paper: https://arxiv.org/abs/2602.12275
#
# This script demonstrates OPCD training for mathematical reasoning:
# - Stage: consolidate (knowledge consolidation)
# - KL type: full (full KL divergence over vocabulary)
# - Top-K: 256 (memory-efficient top-k logit distillation)
# - On-policy: student generates, teacher provides logits with experience
# =============================================================================

# ============ Parse arguments ============
ROLLOUT_N=1
KL_LOSS_TYPE=full
KL_TOPK=256
ACTOR_LR=5e-6
KL_RENORM_TOPK=False
MAX_RESPONSE_LENGTH=16384
REF_MODEL_PATH=""
NNODES=2
EXP_NAME="opcd-math"
EXP_PATH=""
MODEL_PATH="Qwen/Qwen3-8B"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --ref_model_path)
            REF_MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --exp_path)
            EXP_PATH="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --rollout_n)
            ROLLOUT_N="$2"
            shift 2
            ;;
        --kl_loss_type)
            KL_LOSS_TYPE="$2"
            shift 2
            ;;
        --kl_topk)
            KL_TOPK="$2"
            shift 2
            ;;
        --actor_lr)
            ACTOR_LR="$2"
            shift 2
            ;;
        --kl_renorm_topk)
            KL_RENORM_TOPK="$2"
            shift 2
            ;;
        --max_response_length)
            MAX_RESPONSE_LENGTH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

REF_MODEL_PATH=${REF_MODEL_PATH:-$MODEL_PATH}

MAX_PROMPT_LENGTH=$((MAX_RESPONSE_LENGTH + 1024))
PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# ============ Environment ============
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Running OPCD: $EXP_NAME"
echo "Model: $MODEL_PATH"
echo "Ref Model: $REF_MODEL_PATH"
echo "KL Loss Type: $KL_LOSS_TYPE"
echo "KL Top-K: $KL_TOPK"
echo "Experience Path: $EXP_PATH"

# ============ Data Preparation ============
# Prepare DAPO dataset if not already done
if [ ! -f /tmp/dapo_train.parquet ]; then
    echo "Preparing DAPO dataset..."
    python3 -c "
from datasets import load_dataset
import pandas as pd
ds = load_dataset('BytedTsinghua/DAPO', split='train')
ds.to_parquet('/tmp/dapo_train.parquet')
ds_test = load_dataset('BytedTsinghua/DAPO', split='test')
ds_test.to_parquet('/tmp/dapo_test.parquet')
print('DAPO dataset prepared.')
" 2>/dev/null || echo "Warning: Could not auto-download DAPO dataset. Please prepare manually."
fi

# ============ Launch Training ============
python3 -m verl.trainer.main_ppo \
    data.prompt_key=content \
    data.train_files=/tmp/dapo_train.parquet \
    data.val_files=/tmp/dapo_test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=1 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.ref_model_path=$REF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((128 * ROLLOUT_N)) \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.actor.kl_topk=${KL_TOPK} \
    actor_rollout_ref.actor.kl_renorm_topk=${KL_RENORM_TOPK} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.stage=consolidate \
    trainer.experience_path=${EXP_PATH} \
    trainer.on_policy_merge=True \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=OPCD \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=2 \
    trainer.test_freq=10000000000 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_sleep_hack=True \
    trainer.default_local_dir=/tmp/${EXP_NAME} \
    "${@:1}"
