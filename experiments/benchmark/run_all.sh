#!/usr/bin/env bash
# =============================================================================
# EasyOPD Benchmark Experiment Runner
# =============================================================================
# Runs all 12 methods sequentially, saves checkpoints, and evaluates on
# MATH-500 and GSM8K benchmarks.
#
# Usage:
#   bash experiments/benchmark/run_all.sh [method_name]
#   If method_name is provided, only runs that method.
# =============================================================================

set -euo pipefail

export PYTHONPATH="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

# ============ Configuration ============
PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
PROJECT_ROOT="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments/benchmark"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
RESULTS_DIR="${EXPERIMENT_DIR}/results"
LOG_DIR="${EXPERIMENT_DIR}/logs"
DATA_DIR="${EXPERIMENT_DIR}/data"

# Model paths
STUDENT_MODEL="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-1.5B-Instruct"
TEACHER_MODEL="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-7B-Instruct"

# Training hyperparameters
TRAIN_STEPS=200
BATCH_SIZE=16
LR="5e-6"
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=1024
N_GPUS=8
SAVE_FREQ=200  # Save at the end

# Reward function
REWARD_FN="${EXPERIMENT_DIR}/reward_fn.py"

mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${LOG_DIR}" "${DATA_DIR}"

# ============ Helper Functions ============

run_grpo_method() {
    local METHOD_NAME=$1
    local EXTRA_ARGS="${2:-}"
    local CKPT_DIR="${CHECKPOINT_DIR}/${METHOD_NAME}"
    local LOG_FILE="${LOG_DIR}/${METHOD_NAME}.log"

    echo "=============================================="
    echo "Training: ${METHOD_NAME}"
    echo "Checkpoint: ${CKPT_DIR}"
    echo "Log: ${LOG_FILE}"
    echo "=============================================="

    if [ -d "${CKPT_DIR}/global_step_${TRAIN_STEPS}" ] || [ -d "${CKPT_DIR}/actor" ]; then
        echo "Checkpoint already exists, skipping training."
        return 0
    fi

    RAY_ADDRESS=auto ${PYTHON} -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="['${DATA_DIR}/train.parquet']" \
        data.val_files="['${DATA_DIR}/val.parquet']" \
        data.train_batch_size=${BATCH_SIZE} \
        data.max_prompt_length=${MAX_PROMPT_LEN} \
        data.max_response_length=${MAX_RESPONSE_LEN} \
        data.filter_overlong_prompts=True \
        data.truncation=right \
        data.prompt_key=prompt \
        actor_rollout_ref.model.path=${STUDENT_MODEL} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.actor.optim.lr=${LR} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_epochs=2 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.max_model_len=1537 \
        actor_rollout_ref.rollout.max_num_seqs=16 \
        actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        custom_reward_function.path=${REWARD_FN} \
        custom_reward_function.name=compute_score \
        trainer.balance_batch=True \
        'trainer.logger=["console"]' \
        trainer.project_name=easyopd_benchmark \
        trainer.experiment_name=${METHOD_NAME} \
        trainer.n_gpus_per_node=${N_GPUS} \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.total_training_steps=${TRAIN_STEPS} \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.test_freq=50 \
        trainer.default_local_dir=${CKPT_DIR} \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG_FILE}"

    echo "Training ${METHOD_NAME} completed!"
}

run_sft() {
    local CKPT_DIR="${CHECKPOINT_DIR}/sft"
    local LOG_FILE="${LOG_DIR}/sft.log"

    echo "=============================================="
    echo "Training: SFT Baseline"
    echo "=============================================="

    if [ -d "${CKPT_DIR}" ] && [ "$(ls -A ${CKPT_DIR} 2>/dev/null)" ]; then
        echo "SFT checkpoint already exists, skipping."
        return 0
    fi

    ${PYTHON} -m verl.trainer.fsdp_sft_trainer \
        data.train_files="${DATA_DIR}/sft_train.parquet" \
        data.val_files="${DATA_DIR}/val.parquet" \
        data.train_batch_size=64 \
        data.micro_batch_size_per_gpu=2 \
        data.max_length=1536 \
        data.truncation=right \
        data.prompt_key=prompt \
        data.response_key=response \
        model.partial_pretrain=${STUDENT_MODEL} \
        model.enable_gradient_checkpointing=True \
        model.fsdp_config.model_dtype=bf16 \
        optim.lr=2e-5 \
        optim.warmup_steps_ratio=0.05 \
        optim.weight_decay=0.01 \
        use_remove_padding=True \
        trainer.project_name=easyopd_benchmark \
        trainer.experiment_name=sft \
        trainer.total_epochs=3 \
        trainer.total_training_steps=null \
        'trainer.logger=["console"]' \
        trainer.n_gpus_per_node=${N_GPUS} \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.test_freq=50 \
        trainer.default_local_dir=${CKPT_DIR} \
        2>&1 | tee "${LOG_FILE}"

    echo "SFT training completed!"
}

evaluate_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2

    echo "=============================================="
    echo "Evaluating: ${MODEL_NAME}"
    echo "Model: ${MODEL_PATH}"
    echo "=============================================="

    ${PYTHON} "${EXPERIMENT_DIR}/evaluate_model.py" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --output_dir "${RESULTS_DIR}" \
        --benchmarks "math500,gsm8k" \
        --max_tokens 2048 \
        --tensor_parallel_size 1
}

# ============ Main Execution ============

TARGET_METHOD="${1:-all}"

echo "=============================================="
echo "EasyOPD Benchmark Experiment"
echo "Target: ${TARGET_METHOD}"
echo "Student: ${STUDENT_MODEL}"
echo "Teacher: ${TEACHER_MODEL}"
echo "Steps: ${TRAIN_STEPS}"
echo "=============================================="

# Phase 0: Prepare data
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Preparing data..."
    ${PYTHON} "${EXPERIMENT_DIR}/prepare_data.py"
fi

# Phase 0: Evaluate base model
if [ "${TARGET_METHOD}" = "all" ] || [ "${TARGET_METHOD}" = "base" ]; then
    evaluate_model "${STUDENT_MODEL}" "base_qwen2.5-1.5b"
fi

# Phase 1: SFT Baseline
if [ "${TARGET_METHOD}" = "all" ] || [ "${TARGET_METHOD}" = "sft" ]; then
    run_sft
    # Find the latest checkpoint
    SFT_CKPT=$(find "${CHECKPOINT_DIR}/sft" -name "huggingface" -type d 2>/dev/null | sort | tail -1 || echo "")
    if [ -n "${SFT_CKPT}" ]; then
        evaluate_model "${SFT_CKPT}" "sft"
    else
        # Try to find model files directly
        evaluate_model "${CHECKPOINT_DIR}/sft" "sft"
    fi
fi

# Phase 2: GRPO Baseline (no distillation)
if [ "${TARGET_METHOD}" = "all" ] || [ "${TARGET_METHOD}" = "grpo" ]; then
    run_grpo_method "grpo" ""
    evaluate_model "${CHECKPOINT_DIR}/grpo/actor" "grpo"
fi

# Phase 3: Hook-based methods (gkd, sod, opcd, g_opd, sdpo, opsa, ropd, vision_opd)
HOOK_METHODS="gkd sod opcd g_opd sdpo opsa ropd vision_opd"
for METHOD in ${HOOK_METHODS}; do
    if [ "${TARGET_METHOD}" = "all" ] || [ "${TARGET_METHOD}" = "${METHOD}" ]; then
        run_grpo_method "${METHOD}" "+easyopd.method.name=${METHOD}"
        evaluate_model "${CHECKPOINT_DIR}/${METHOD}/actor" "${METHOD}"
    fi
done

# Phase 4: Cross-tokenizer methods (simple, simct)
# These use use_kl_in_reward=True to add KL divergence as reward signal
for METHOD in simple simct; do
    if [ "${TARGET_METHOD}" = "all" ] || [ "${TARGET_METHOD}" = "${METHOD}" ]; then
        EXTRA="algorithm.use_kl_in_reward=True ++actor_rollout_ref.actor.policy_loss.loss_mode=${METHOD} ++actor_rollout_ref.actor.policy_loss.simple_kl_direction=reverse ++actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0"
        run_grpo_method "${METHOD}" "${EXTRA}"
        evaluate_model "${CHECKPOINT_DIR}/${METHOD}/actor" "${METHOD}"
    fi
done

# Phase 5: Generate summary table
echo ""
echo "=============================================="
echo "Generating summary table..."
echo "=============================================="
${PYTHON} "${EXPERIMENT_DIR}/generate_table.py"

echo ""
echo "ALL EXPERIMENTS COMPLETED!"
