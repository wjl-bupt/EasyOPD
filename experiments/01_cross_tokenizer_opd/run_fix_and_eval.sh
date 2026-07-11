#!/bin/bash
# ==============================================================================
# run_fix_and_eval.sh — Fix script to complete remaining work:
#   1. Merge ALM FSDP checkpoints to HF format (if not done)
#   2. Evaluate ULD checkpoints (4 benchmarks)
#   3. Evaluate ALM checkpoints (4 benchmarks)
#   4. Retrain DSKD with OOM fix, then merge + evaluate
#
# Usage:
#   nohup bash run_fix_and_eval.sh > /path/to/workspace/eval_logs/run_fix_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ==============================================================================

set -u

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/path/to/workspace/eval_logs"
mkdir -p "${LOG_DIR}"

EXP_DIR="/path/to/EasyOPD/experiments/01_cross_tokenizer_opd"
RUNS_ROOT="/path/to/models/runs"
EXP_NAME="01_cross_tokenizer_opd"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ==============================================================================
# Phase 1: Wait for current ALM training to finish (if still running)
# ==============================================================================
log "Phase 1: Checking if ALM training is still running..."
while ps aux | grep -q "[T]askRunner.run"; do
    log "  ALM training still in progress, waiting 60s..."
    sleep 60
done
log "  No active training detected. Proceeding."

# Stop Ray to free GPU memory
ray stop --force 2>/dev/null || true
sleep 10

# ==============================================================================
# Phase 2: Merge ALM FSDP checkpoints to HF (if not already done)
# ==============================================================================
log "Phase 2: Merging ALM checkpoints..."
ALM_METHOD_DIR="${EXP_DIR}/methods/alm"
(
    cd "${ALM_METHOD_DIR}"
    # Source the variables we need from launch.sh without running the full script
    export PYTHONPATH="/path/to/EasyOPD:${PYTHONPATH:-}"
    PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
    SHARED_SCRIPTS="/path/to/EasyOPD/experiments/_shared/scripts"
    STUDENT_MODEL="/path/to/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini/hf/global_step_116"
    FSDP_CKPT_DIR="/path/to/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/fsdp"
    HF_CKPT_DIR="/path/to/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf"
    mkdir -p "${HF_CKPT_DIR}"

    shopt -s nullglob
    ALL_CKPTS=( "${FSDP_CKPT_DIR}"/global_step_* )
    shopt -u nullglob

    for CKPT_DIR in "${ALL_CKPTS[@]}"; do
        STEP_NAME=$(basename "${CKPT_DIR}")
        ACTOR_DIR="${CKPT_DIR}/actor"
        TARGET_DIR="${HF_CKPT_DIR}/${STEP_NAME}"

        if [ ! -d "${ACTOR_DIR}" ]; then
            log "  [${STEP_NAME}] No actor/ subdir, skipping."
            continue
        fi
        if [ -f "${TARGET_DIR}/model.safetensors" ] || [ -f "${TARGET_DIR}/model.safetensors.index.json" ]; then
            log "  [${STEP_NAME}] Already merged, skipping."
            continue
        fi

        log "  [${STEP_NAME}] Merging..."
        ${PYTHON} ${SHARED_SCRIPTS}/merge_fsdp.py \
            --ckpt_dir "${ACTOR_DIR}" \
            --base_model "${STUDENT_MODEL}" \
            --output_dir "${TARGET_DIR}" \
            2>&1 | tail -3
    done
)
log "Phase 2: ALM merge done."

# ==============================================================================
# Phase 3: Evaluate ULD + ALM (all 4 benchmarks per checkpoint)
# ==============================================================================
log "Phase 3: Evaluating ULD and ALM checkpoints..."

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
EVAL_SGLANG_SCRIPT="/path/to/EasyOPD/experiments/_shared/scripts/evaluate_model_sglang.py"
EVAL_PORT=30000
EVAL_BASE_URL="http://127.0.0.1:${EVAL_PORT}"
EVAL_TEMPERATURE=0.6
EVAL_TOP_P=0.95
EVAL_MAX_CONCURRENT=256
EVAL_DATA_DIR="/path/to/EasyOPD/experiments/_shared/eval_data"

declare -A BENCH_TOKENS
BENCH_TOKENS[math500]=4096
BENCH_TOKENS[gsm8k]=4096
BENCH_TOKENS[mbpp]=2048
BENCH_TOKENS[live-code-bench-v6]=4096

ALL_BENCHMARKS=("math500" "gsm8k" "mbpp" "live-code-bench-v6")

eval_kill_server() {
    local pids=$(ps aux | grep -E "[v]llm.entrypoints.openai" | grep "\-\-port ${EVAL_PORT}" | awk '{print $2}' || true)
    [ -n "$pids" ] && { echo "$pids" | xargs kill -9 2>/dev/null; sleep 3; } || true
    pids=$(ps aux | grep -E "multiprocessing\.(spawn|resource_tracker)" | grep -v grep | awk '{print $2}' || true)
    [ -n "$pids" ] && { echo "$pids" | xargs kill -9 2>/dev/null; sleep 1; } || true
    return 0
}

eval_start_server() {
    local model_path="$1" gpu_id="$2"
    log "  Starting vLLM server on GPU ${gpu_id}: $(basename ${model_path})"

    CUDA_VISIBLE_DEVICES=${gpu_id} ${PYTHON} -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" --port ${EVAL_PORT} --trust-remote-code \
        --gpu-memory-utilization 0.85 --max-model-len 16384 \
        --disable-log-requests --enforce-eager --seed 42 \
        > /tmp/vllm_eval_fix.log 2>&1 &
    local spid=$!

    for i in $(seq 1 60); do
        sleep 5
        if ! kill -0 $spid 2>/dev/null; then
            log "  Server died!"; tail -5 /tmp/vllm_eval_fix.log; return 1
        fi
        if curl -s http://127.0.0.1:${EVAL_PORT}/v1/models 2>/dev/null | grep -q "data"; then
            log "  Server ready!"; return 0
        fi
    done
    log "  Timeout!"; return 1
}

eval_method() {
    local method="$1"
    local run_name="$2"
    local hf_dir="${RUNS_ROOT}/${EXP_NAME}/${method}/${run_name}/hf"
    local results_dir="${EXP_DIR}/methods/${method}/results/${run_name}"
    mkdir -p "${results_dir}"

    log "  === Evaluating ${method^^} ==="

    shopt -s nullglob
    local ckpts=( "${hf_dir}"/global_step_* )
    shopt -u nullglob

    if [ ${#ckpts[@]} -eq 0 ]; then
        log "  No HF checkpoints found in ${hf_dir}, skipping."
        return 0
    fi

    local gpu_id=0
    for merged_dir in "${ckpts[@]}"; do
        local step_name=$(basename "${merged_dir}")
        local tag="${run_name}_${step_name}"

        if [ ! -f "${merged_dir}/model.safetensors" ] && [ ! -f "${merged_dir}/model.safetensors.index.json" ]; then
            log "  [${step_name}] Not a valid HF checkpoint, skip."
            continue
        fi

        # Check if all benchmarks already done
        local all_done=true
        for bench in "${ALL_BENCHMARKS[@]}"; do
            [ ! -f "${results_dir}/${tag}_${bench}_details.json" ] && { all_done=false; break; }
        done
        if [ "${all_done}" = true ]; then
            log "  [${step_name}] All benchmarks done, skip."
            continue
        fi

        # Start vLLM server
        eval_kill_server
        if ! eval_start_server "${merged_dir}" "${gpu_id}"; then
            log "  [${step_name}] First attempt failed, retrying..."
            eval_kill_server; sleep 5
            if ! eval_start_server "${merged_dir}" "${gpu_id}"; then
                log "  [${step_name}] FAILED to start server, skip."
                eval_kill_server; continue
            fi
        fi

        # Evaluate all benchmarks
        for bench in "${ALL_BENCHMARKS[@]}"; do
            local output_dir="${results_dir}/${tag}_${bench}"
            local details_json="${results_dir}/${tag}_${bench}_details.json"

            if [ -f "${details_json}" ]; then
                log "  [${step_name}] ${bench}: already done, skip."
                continue
            fi

            mkdir -p "${output_dir}"
            log "  [${step_name}] Evaluating ${bench}..."
            ${PYTHON} ${EVAL_SGLANG_SCRIPT} --model_path "${merged_dir}" --dataset "${bench}" \
                --base_url "${EVAL_BASE_URL}" --output_dir "${output_dir}" \
                --data_dir "${EVAL_DATA_DIR}" \
                --max_new_tokens ${BENCH_TOKENS[$bench]} --max_concurrent ${EVAL_MAX_CONCURRENT} \
                --temperature ${EVAL_TEMPERATURE} --top_p ${EVAL_TOP_P} \
                2>&1 | tail -5
            if [ $? -eq 0 ] && [ -f "${output_dir}/metrics.json" ]; then
                cp "${output_dir}/metrics.json" "${details_json}"
                log "  [${step_name}] ${bench}: Done"
            else
                log "  [${step_name}] ${bench}: Failed"
            fi
        done

        eval_kill_server; sleep 5
        gpu_id=$(( (gpu_id + 1) % 8 ))
    done
    return 0
}

eval_kill_server
eval_method "uld" "uld_phi4mini"
eval_method "alm" "alm_phi4mini"
log "Phase 3: Evaluation done."

# ==============================================================================
# Phase 4: Retrain DSKD with OOM fix
# ==============================================================================
log "Phase 4: Retraining DSKD..."
eval_kill_server

DSKD_METHOD_DIR="${EXP_DIR}/methods/dskd"
DSKD_LOCAL_DIR="${RUNS_ROOT}/${EXP_NAME}/dskd/dskd_phi4mini"
DSKD_NET_CKPT="${EXP_DIR}/methods/dskd/checkpoints"

# Clean old DSKD checkpoints
if [ -d "${DSKD_LOCAL_DIR}" ]; then
    log "  Removing old DSKD local dir: ${DSKD_LOCAL_DIR}"
    rm -rf "${DSKD_LOCAL_DIR}"
fi
if [ -d "${DSKD_NET_CKPT}" ]; then
    log "  Removing old DSKD network ckpt: ${DSKD_NET_CKPT}"
    rm -rf "${DSKD_NET_CKPT}"
fi

# Train DSKD with safe mode (includes OOM fix: PPO_MAX_TOKEN=12288)
log "  Starting DSKD training..."
DSKD_LOG="${LOG_DIR}/train_dskd_fix_${TIMESTAMP}.log"
(
    cd "${DSKD_METHOD_DIR}" && \
    SPEED_TIER=safe bash launch.sh
) > "${DSKD_LOG}" 2>&1
DSKD_RC=$?

if [ ${DSKD_RC} -ne 0 ]; then
    log "  DSKD training FAILED, rc=${DSKD_RC}. Check: ${DSKD_LOG}"
else
    log "  DSKD training completed successfully."
fi

# ==============================================================================
# Phase 5: Evaluate DSKD (in case eval was skipped due to the old bug)
# ==============================================================================
if [ ${DSKD_RC} -eq 0 ]; then
    log "Phase 5: Verifying DSKD evaluation..."
    eval_kill_server
    eval_method "dskd" "dskd_phi4mini"
    log "Phase 5: DSKD evaluation done."
fi

# ==============================================================================
# Final Summary
# ==============================================================================
log ""
log "=========================================="
log "=== ALL DONE ==="
log "=========================================="
log "ULD results:  ${EXP_DIR}/methods/uld/results/uld_phi4mini/"
log "DSKD results: ${EXP_DIR}/methods/dskd/results/dskd_phi4mini/"
log "ALM results:  ${EXP_DIR}/methods/alm/results/alm_phi4mini/"
log "=========================================="
