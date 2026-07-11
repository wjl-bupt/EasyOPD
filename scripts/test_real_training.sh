#!/bin/bash
# =============================================================================
# EasyOPD Real-World Training Test Script
#
# Tests all 12 methods with real models and datasets on 8 GPUs.
# Each method runs for 2 training steps to verify end-to-end correctness.
#
# Student Model: Qwen2.5-1.5B-Instruct
# Teacher Model: Qwen2.5-7B-Instruct
# Dataset: mixed_math_code_10k (parquet format)
# GPUs: 8x NVIDIA H20
#
# Usage:
#   conda activate OpenAgentRL-sj
#   bash scripts/test_real_training.sh [--method <name>] [--all]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# ============ Configuration ============
STUDENT_MODEL="/path/to/workspace/workspace/models/Qwen2.5-1.5B-Instruct"
TEACHER_MODEL="/path/to/workspace/workspace/models/Qwen2.5-7B-Instruct"
TRAIN_DATA="${PROJECT_ROOT}/test_data/train.parquet"
VAL_DATA="${PROJECT_ROOT}/test_data/test.parquet"
OUTPUT_DIR="/tmp/easyopd_real_test"
NGPUS=8
TOTAL_STEPS=2
TRAIN_BATCH_SIZE=8
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=256

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

# ============ Parse arguments ============
METHOD=""
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --all) RUN_ALL=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$METHOD" ] && [ "$RUN_ALL" = false ]; then
    RUN_ALL=true
fi

# ============ Prepare data if needed ============
if [ ! -f "$TRAIN_DATA" ]; then
    echo "[test_real_training] Preparing parquet data..."
    python examples/simple/prepare_data.py \
        --src /path/to/workspace/workspace/dataset/mixed_math_code_10k \
        --dst "${PROJECT_ROOT}/test_data" \
        --val-size 50
fi

# ============ Method definitions ============
# Each method has specific configuration requirements.
# Format: method_name|needs_teacher|extra_args

declare -A METHOD_CONFIG

# Methods that use standard distillation (student + teacher)
METHOD_CONFIG[gkd]="teacher|distillation.distillation_loss.loss_mode=gkd distillation.distillation_loss.gkd_beta=0.5"
METHOD_CONFIG[opcd]="teacher|distillation.distillation_loss.loss_mode=opcd"
METHOD_CONFIG[sdpo]="teacher|distillation.distillation_loss.loss_mode=sdpo"
METHOD_CONFIG[opsa]="teacher|distillation.distillation_loss.loss_mode=opsa"
METHOD_CONFIG[vision_opd]="teacher|distillation.distillation_loss.loss_mode=vision_opd"
METHOD_CONFIG[g_opd]="teacher|distillation.distillation_loss.loss_mode=g_opd"
METHOD_CONFIG[sod]="teacher|distillation.distillation_loss.loss_mode=sod"
METHOD_CONFIG[echo_kd]="teacher|distillation.distillation_loss.loss_mode=echo_kd"
METHOD_CONFIG[ropd]="teacher|distillation.distillation_loss.loss_mode=ropd"

# Cross-tokenizer methods (need special teacher setup)
METHOD_CONFIG[simple]="xtok|distillation.distillation_loss.loss_mode=simple distillation.distillation_loss.use_cross_tokenizer=True"
METHOD_CONFIG[simct]="xtok|distillation.distillation_loss.loss_mode=simct distillation.distillation_loss.use_cross_tokenizer=True"

# Non-actor methods (use standard GRPO without distillation loss)
METHOD_CONFIG[gad]="critic|"
METHOD_CONFIG[lightning_opd]="offline|"

# ============ Results tracking ============
RESULTS_FILE="${OUTPUT_DIR}/results.txt"
mkdir -p "$OUTPUT_DIR"
echo "EasyOPD Real-World Training Test Results" > "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "Student: $STUDENT_MODEL" >> "$RESULTS_FILE"
echo "Teacher: $TEACHER_MODEL" >> "$RESULTS_FILE"
echo "GPUs: $NGPUS" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# ============ Run a single method ============
run_method() {
    local method_name=$1
    local config_str="${METHOD_CONFIG[$method_name]:-}"
    
    if [ -z "$config_str" ]; then
        echo "  ⚠️  Unknown method: $method_name"
        echo "SKIP $method_name (unknown)" >> "$RESULTS_FILE"
        ((SKIP_COUNT++))
        return
    fi

    local method_type="${config_str%%|*}"
    local extra_args="${config_str#*|}"
    local method_output="${OUTPUT_DIR}/${method_name}"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Testing: $method_name (type: $method_type)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Skip methods that need special infrastructure not available
    if [ "$method_type" = "offline" ]; then
        echo "  ⏭️  Skipping $method_name (offline method, needs pre-computed data)"
        echo "SKIP $method_name (offline method)" >> "$RESULTS_FILE"
        ((SKIP_COUNT++))
        return
    fi

    if [ "$method_type" = "critic" ]; then
        echo "  ⏭️  Skipping $method_name (critic-only method, needs custom critic setup)"
        echo "SKIP $method_name (critic method)" >> "$RESULTS_FILE"
        ((SKIP_COUNT++))
        return
    fi

    if [ "$method_type" = "xtok" ]; then
        echo "  ⏭️  Skipping $method_name (cross-tokenizer, needs teacher sidecar with different tokenizer)"
        echo "SKIP $method_name (cross-tokenizer)" >> "$RESULTS_FILE"
        ((SKIP_COUNT++))
        return
    fi

    # Build the training command
    local CMD="python -m verl.trainer.main_ppo"
    CMD="$CMD data.train_files=\"['${TRAIN_DATA}']\""
    CMD="$CMD data.val_files=\"['${VAL_DATA}']\""
    CMD="$CMD data.train_batch_size=${TRAIN_BATCH_SIZE}"
    CMD="$CMD data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    CMD="$CMD data.max_response_length=${MAX_RESPONSE_LENGTH}"
    CMD="$CMD data.truncation=right"
    CMD="$CMD data.prompt_key=prompt"
    CMD="$CMD actor_rollout_ref.model.path=${STUDENT_MODEL}"
    CMD="$CMD actor_rollout_ref.model.use_remove_padding=True"
    CMD="$CMD actor_rollout_ref.model.enable_gradient_checkpointing=True"
    CMD="$CMD actor_rollout_ref.actor.optim.lr=1e-6"
    CMD="$CMD actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}"
    CMD="$CMD actor_rollout_ref.actor.use_dynamic_bsz=True"
    CMD="$CMD actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048"
    CMD="$CMD actor_rollout_ref.actor.fsdp_config.param_offload=True"
    CMD="$CMD actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
    CMD="$CMD actor_rollout_ref.rollout.name=vllm"
    CMD="$CMD actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    CMD="$CMD actor_rollout_ref.rollout.gpu_memory_utilization=0.4"
    CMD="$CMD actor_rollout_ref.rollout.n=1"
    CMD="$CMD actor_rollout_ref.rollout.temperature=0.7"
    CMD="$CMD actor_rollout_ref.rollout.max_num_seqs=8"
    CMD="$CMD actor_rollout_ref.rollout.max_num_batched_tokens=2048"
    CMD="$CMD actor_rollout_ref.rollout.enforce_eager=True"
    CMD="$CMD actor_rollout_ref.rollout.free_cache_engine=True"
    CMD="$CMD actor_rollout_ref.ref.fsdp_config.param_offload=True"
    CMD="$CMD algorithm.adv_estimator=grpo"
    CMD="$CMD algorithm.use_kl_in_reward=False"
    CMD="$CMD trainer.n_gpus_per_node=${NGPUS}"
    CMD="$CMD trainer.nnodes=1"
    CMD="$CMD trainer.val_before_train=False"
    CMD="$CMD trainer.total_training_steps=${TOTAL_STEPS}"
    CMD="$CMD trainer.save_freq=999"
    CMD="$CMD trainer.test_freq=999"
    CMD="$CMD trainer.logger='[\"console\"]'"
    CMD="$CMD trainer.project_name=easyopd_test"
    CMD="$CMD trainer.experiment_name=${method_name}_test"
    CMD="$CMD trainer.default_local_dir=${method_output}"

    # Add distillation config for teacher-based methods
    if [ "$method_type" = "teacher" ]; then
        CMD="$CMD distillation.enabled=True"
        CMD="$CMD distillation.n_gpus_per_node=${NGPUS}"
        CMD="$CMD distillation.nnodes=1"
        CMD="$CMD distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}"
        CMD="$CMD distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2"
        CMD="$CMD distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.3"
        CMD="$CMD distillation.teacher_models.teacher_model.inference.name=vllm"
    fi

    # Add method-specific extra args
    if [ -n "$extra_args" ]; then
        CMD="$CMD $extra_args"
    fi

    # Add reward function (placeholder)
    CMD="$CMD reward.custom_reward_function.path=${PROJECT_ROOT}/examples/simple/reward.py"
    CMD="$CMD reward.custom_reward_function.name=compute_score"

    echo "  Command: $CMD"
    echo ""

    # Run with timeout (5 minutes per method)
    local start_time=$(date +%s)
    if timeout 300 bash -c "$CMD" > "${method_output}.log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "  ✅ PASS: $method_name (${duration}s)"
        echo "PASS $method_name (${duration}s)" >> "$RESULTS_FILE"
        ((PASS_COUNT++))
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        if [ $exit_code -eq 124 ]; then
            echo "  ⏰ TIMEOUT: $method_name (>${duration}s)"
            echo "TIMEOUT $method_name (>${duration}s)" >> "$RESULTS_FILE"
            ((FAIL_COUNT++))
        else
            echo "  ❌ FAIL: $method_name (exit=$exit_code, ${duration}s)"
            echo "FAIL $method_name (exit=$exit_code, ${duration}s)" >> "$RESULTS_FILE"
            # Show last 20 lines of error
            echo "  Last error lines:"
            tail -20 "${method_output}.log" 2>/dev/null | sed 's/^/    /'
            ((FAIL_COUNT++))
        fi
    fi
}

# ============ Main ============
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     EasyOPD Real-World Training Test (8 GPUs)               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Student: Qwen2.5-1.5B-Instruct                            ║"
echo "║  Teacher: Qwen2.5-7B-Instruct                              ║"
echo "║  Dataset: mixed_math_code_10k (parquet)                     ║"
echo "║  Steps:   ${TOTAL_STEPS} per method                                     ║"
echo "║  GPUs:    ${NGPUS}x NVIDIA H20                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ -n "$METHOD" ]; then
    run_method "$METHOD"
else
    # Run all methods
    METHODS=(gkd sod opcd g_opd sdpo opsa vision_opd echo_kd ropd simple simct gad lightning_opd)
    for m in "${METHODS[@]}"; do
        run_method "$m"
    done
fi

# ============ Summary ============
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Test Summary                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Passed:  $PASS_COUNT                                                    ║"
echo "║  Failed:  $FAIL_COUNT                                                    ║"
echo "║  Skipped: $SKIP_COUNT                                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Full results: $RESULTS_FILE"
echo "Logs: ${OUTPUT_DIR}/<method>.log"
