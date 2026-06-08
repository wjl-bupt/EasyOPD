#!/bin/bash
# Monitor Simple training and auto-start SimCT after completion
set -euo pipefail

LOG="/tmp/easyopd_simple_phi4mini_final.log"
CKPT_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/benchmark/checkpoints/simple_phi4mini"
PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
WORKDIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD"

echo "[$(date)] Monitoring Simple training..."

# Wait for Simple training to finish
while true; do
    # Check if process is still running
    if ! ps aux | grep "main_ppo" | grep -v grep > /dev/null 2>&1; then
        echo "[$(date)] Simple training process ended."
        break
    fi
    
    # Check progress
    progress=$(grep "Training Progress" "$LOG" | tail -1)
    echo "[$(date)] $progress"
    
    # Check if 100% or completed
    if echo "$progress" | grep -q "100%"; then
        echo "[$(date)] Simple training completed!"
        break
    fi
    
    sleep 120
done

# Check for errors
if grep -q "Traceback" "$LOG"; then
    echo "[$(date)] ERROR: Simple training failed with traceback!"
    grep "Traceback" "$LOG" | tail -3
    exit 1
fi

echo "[$(date)] Simple training finished successfully."

# Step 1: Merge FSDP checkpoint
echo "[$(date)] Merging FSDP checkpoint..."
CKPT_STEP=$(ls -d ${CKPT_DIR}/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -z "$CKPT_STEP" ]; then
    echo "[$(date)] ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "[$(date)] Found checkpoint: $CKPT_STEP"
MERGED_DIR="${CKPT_DIR}/merged_hf"
mkdir -p "$MERGED_DIR"

cd "$WORKDIR"
$PYTHON -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, glob

ckpt_dir = '${CKPT_STEP}/actor'
merged_dir = '${MERGED_DIR}'
base_model = '/root/workspace/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6/checkpoint-40'

print(f'Loading base model from {base_model}...')
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Load FSDP sharded state dict
print(f'Loading FSDP checkpoint from {ckpt_dir}...')
shard_files = sorted(glob.glob(os.path.join(ckpt_dir, 'model_*.pt')))
if not shard_files:
    # Try huggingface format
    shard_files = sorted(glob.glob(os.path.join(ckpt_dir, '*.safetensors')))
    if shard_files:
        from safetensors.torch import load_file
        state_dict = {}
        for f in shard_files:
            state_dict.update(load_file(f))
    else:
        raise FileNotFoundError(f'No checkpoint files found in {ckpt_dir}')
else:
    state_dict = {}
    for f in shard_files:
        state_dict.update(torch.load(f, map_location='cpu'))

print(f'Loaded {len(state_dict)} parameters')
model.load_state_dict(state_dict, strict=False)

print(f'Saving merged model to {merged_dir}...')
model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print('Merge complete!')
" 2>&1

echo "[$(date)] Merge complete. Starting SimCT training..."

# Step 2: Restart Ray and launch SimCT
/opt/conda/envs/OpenAgentRL-sj/bin/ray stop --force 2>/dev/null
sleep 5
/opt/conda/envs/OpenAgentRL-sj/bin/ray start --head --disable-usage-stats --num-cpus=64 --num-gpus=8 --include-dashboard=false 2>&1 | tail -2

sleep 10

# Launch SimCT training
cd "$WORKDIR"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export RAY_ADDRESS=auto

SIMCT_CKPT_DIR="${WORKDIR}/experiments/benchmark/checkpoints/simct_phi4mini"
mkdir -p "$SIMCT_CKPT_DIR"

echo "[$(date)] Launching SimCT training..."
$PYTHON -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    "data.train_files=['${WORKDIR}/experiments/benchmark/data_phi4mini/train.parquet']" \
    "data.val_files=['${WORKDIR}/experiments/benchmark/data_phi4mini/val.parquet']" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path="/root/workspace/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6/checkpoint-40" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    "++actor_rollout_ref.actor.policy_loss.loss_mode=simct" \
    "++actor_rollout_ref.actor.policy_loss.simple_kl_direction=reverse" \
    "++actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0" \
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
    custom_reward_function.path="${WORKDIR}/experiments/benchmark/reward_fn.py" \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    'trainer.logger=["console"]' \
    trainer.project_name=easyopd_benchmark \
    trainer.experiment_name=simct_phi4mini \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=200 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.default_local_dir="${SIMCT_CKPT_DIR}" 2>&1

echo "[$(date)] SimCT training finished."
