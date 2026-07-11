#!/bin/bash
set -euo pipefail
trap 'rc=$?; echo "[FATAL] launch.sh exited with code $rc at line $LINENO (last cmd: $BASH_COMMAND)" >&2' ERR

# ============================================================
# GRPO Baseline — Agentic OPD Experiment
#
# Pure GRPO without SOD's token-level KL regularization.
# This serves as the baseline to compare against SOD.
#
# Key differences from SOD:
#   - No token_kl_reg (SOD's core contribution removed)
#   - No separate teacher model (ref = student itself)
#   - kl_coef = 0.0 (no KL penalty)
#   - All other hyperparameters identical for fair comparison
#
# Training Config:
#   - Model: Qwen3-1.7B-SFT (global_step_115)
#   - Dataset: Open-AgentRL-30K (multi-turn agent interactions)
#   - Hardware: 8x NVIDIA H20 (96GB)
#   - Batch size: 64, rollout n=16
#   - infer_tp=4, train_sp=4
#   - save_freq=10 -> checkpoints at step 10, 20, 30, ...
#   - total_epochs=1
# ============================================================

# ============ Environment Setup ============
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1  # Required: enables vLLM V1 engine for async rollout
# Load local secrets/paths if present (.env at repo root, git-ignored).
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
[ -f "${_REPO_ROOT}/.env" ] && set -a && . "${_REPO_ROOT}/.env" && set +a
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-https://YOUR_SANDBOX_FUSION_ENDPOINT/run_code}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PYTHON="/opt/conda/envs/OpenAgentRL/bin/python"
RAY="/opt/conda/envs/OpenAgentRL/bin/ray"
export PATH="/opt/conda/envs/OpenAgentRL/bin:${PATH}"

# ----------------- Paths -----------------
EASYOPD_ROOT="/path/to/workspace/SOD_Merge_Framework/EasyOPD_clone"
EXP_DIR="${EASYOPD_ROOT}/experiments/02_agentic_opd"
METHOD_DIR="${EXP_DIR}/methods/grpo"
RESULTS_DIR="${METHOD_DIR}/results"
CKPT_DIR="${METHOD_DIR}/checkpoints"
mkdir -p "${RESULTS_DIR}" "${CKPT_DIR}"

# Model: Qwen3-1.7B-SFT (same starting point as SOD)
MODEL="/path/to/workspace/checkpoint/qwen3_1p7b_sft/global_step_115/huggingface"

# Dataset (same as SOD)
TRAIN_DATA="/path/to/workspace/dataset/Gen-Verse/Open-AgentRL-30K/Open-AgentRL-30K.parquet"
VAL_DATA_1="/path/to/workspace/dataset/Gen-Verse/Open-AgentRL-Eval/aime2025/aime_2025_problems.parquet"
VAL_DATA_2="/path/to/workspace/dataset/Gen-Verse/Open-AgentRL-Eval/aime2024/aime_2024_problems.parquet"

# Output
SAVE_DIR="${CKPT_DIR}/grpo_training"
PROJECT_NAME="grpo_agentic_opd"
EXPERIMENT_NAME="grpo_qwen3_1p7b"

# Training hyperparameters (same as SOD for fair comparison)
SAVE_FREQ=10
TOTAL_EPOCHS=1

echo "[$(date)] ===== GRPO Baseline Experiment ====="
echo "[$(date)] Model: ${MODEL}"
echo "[$(date)] Train data: ${TRAIN_DATA}"
echo "[$(date)] Save dir: ${SAVE_DIR}"

# ===== Step 1: Install EasyOPD =====
echo "[$(date)] ===== Step 1: Installing EasyOPD ====="
pushd "${EASYOPD_ROOT}" > /dev/null
${PYTHON} -m pip install -e . --quiet 2>&1 | tail -5
popd > /dev/null

# ===== Step 2: Start Ray =====
echo "[$(date)] ===== Step 2: Starting Ray cluster ====="
${RAY} stop --force 2>/dev/null || true
sleep 3

# ===== Step 3: GRPO Training =====
echo "[$(date)] ===== Step 3: Starting GRPO Training ====="

pushd "${EASYOPD_ROOT}" > /dev/null

${PYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    +algorithm.filter_groups.enable=False \
    +algorithm.filter_groups.metric=seq_reward \
    +algorithm.filter_groups.reward_std_threshold=0.0 \
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
    actor_rollout_ref.model.path=${MODEL} \
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
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.test_freq=${SAVE_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    2>&1 | tee "${CKPT_DIR}/training.log"

popd > /dev/null

echo "[$(date)] ===== Step 3 Complete: Training finished ====="

# ===== Step 4: Merge FSDP checkpoints to HF format =====
echo "[$(date)] ===== Step 4: Merging checkpoints to HF format ====="

FSDP_CKPT_DIR="${SAVE_DIR}"
HF_CKPT_DIR="${CKPT_DIR}/hf"
mkdir -p "${HF_CKPT_DIR}"

# Find all global_step_* directories
ALL_CKPTS=()
for ckpt_dir in "${FSDP_CKPT_DIR}"/global_step_*/actor/; do
    if [ -d "${ckpt_dir}" ]; then
        step_name=$(basename "$(dirname "${ckpt_dir}")")
        ALL_CKPTS+=("${step_name}")
    fi
done

if [ ${#ALL_CKPTS[@]} -eq 0 ]; then
    echo "[$(date)] WARNING: No FSDP checkpoints found in ${FSDP_CKPT_DIR}/global_step_*/actor/"
    echo "[$(date)] Trying alternative checkpoint structure..."
    for ckpt_dir in "${FSDP_CKPT_DIR}"/global_step_*/; do
        if [ -d "${ckpt_dir}" ]; then
            step_name=$(basename "${ckpt_dir}")
            ALL_CKPTS+=("${step_name}")
        fi
    done
fi

echo "[$(date)] Found ${#ALL_CKPTS[@]} checkpoint(s): ${ALL_CKPTS[*]:-none}"

for step_name in "${ALL_CKPTS[@]}"; do
    TARGET_DIR="${HF_CKPT_DIR}/${step_name}"
    if [ -d "${TARGET_DIR}" ] && [ -f "${TARGET_DIR}/config.json" ]; then
        echo "[$(date)] Skipping ${step_name} (already merged)"
        continue
    fi
    echo "[$(date)] Merging ${step_name} -> ${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}"

    # Try FSDP merge script if available
    if [ -f "${EASYOPD_ROOT}/scripts/merge_fsdp_checkpoint.py" ]; then
        ${PYTHON} "${EASYOPD_ROOT}/scripts/merge_fsdp_checkpoint.py" \
            --fsdp_checkpoint_dir "${FSDP_CKPT_DIR}/${step_name}/actor" \
            --model_path "${MODEL}" \
            --output_dir "${TARGET_DIR}" \
            2>&1 || echo "[$(date)] WARNING: merge script failed for ${step_name}, trying alternative..."
    fi

    # If merge didn't produce config.json, try verl's built-in merge
    if [ ! -f "${TARGET_DIR}/config.json" ]; then
        if [ -f "${EASYOPD_ROOT}/verl/utils/fsdp_utils.py" ]; then
            ${PYTHON} -c "
from verl.utils.fsdp_utils import merge_fsdp_checkpoint
merge_fsdp_checkpoint(
    fsdp_checkpoint_path='${FSDP_CKPT_DIR}/${step_name}/actor',
    hf_model_path='${MODEL}',
    output_path='${TARGET_DIR}'
)
print('Merge successful: ${step_name}')
" 2>&1 || echo "[$(date)] WARNING: verl merge also failed for ${step_name}"
        fi
    fi
done

echo "[$(date)] ===== Step 4 Complete: Checkpoint merging done ====="

# ===== Step 5: Generate summary =====
echo "[$(date)] ===== Step 5: Generating results summary ====="

${PYTHON} -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
ckpt_dir = '${CKPT_DIR}'
summary = {
    'experiment': 'GRPO Baseline (Agentic OPD)',
    'model': 'Qwen3-1.7B-SFT (global_step_115)',
    'teacher_model': 'None (pure GRPO, no distillation)',
    'dataset': 'Open-AgentRL-30K',
    'eval_datasets': ['AIME 2024', 'AIME 2025'],
    'note': 'Baseline for comparison with SOD. All hyperparameters identical except no token_kl_reg and no teacher model.',
    'checkpoints': []
}

hf_dir = os.path.join(ckpt_dir, 'hf')
if os.path.exists(hf_dir):
    for d in sorted(os.listdir(hf_dir)):
        if d.startswith('global_step_'):
            full_path = os.path.join(hf_dir, d)
            has_config = os.path.exists(os.path.join(full_path, 'config.json'))
            summary['checkpoints'].append({
                'step': d,
                'path': full_path,
                'merged': has_config
            })

with open(os.path.join(results_dir, 'grpo_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved with {len(summary[\"checkpoints\"])} checkpoint(s)')
" 2>&1

echo "[$(date)] ===== GRPO Baseline Experiment Complete ====="
echo "[$(date)] Results: ${RESULTS_DIR}"
echo "[$(date)] Checkpoints: ${CKPT_DIR}"
