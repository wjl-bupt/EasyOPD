#!/bin/bash
set -euo pipefail

# ==============================================================================
# DSKD-only: Train + Merge + Evaluate (all 4 benchmarks)
# ==============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/path/to/workspace/eval_logs"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

EXP_DIR="/path/to/EasyOPD/experiments/01_cross_tokenizer_opd"
RUNS_ROOT="/path/to/models/runs"
EXP_NAME="01_cross_tokenizer_opd"
DSKD_LOCAL_DIR="${RUNS_ROOT}/${EXP_NAME}/dskd/dskd_phi4mini"
DSKD_METHOD_DIR="${EXP_DIR}/methods/dskd"
RESULTS_DIR="${EXP_DIR}/methods/dskd/results/dskd_phi4mini"

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
EVAL_PORT=30000
EVAL_BASE_URL="http://127.0.0.1:${EVAL_PORT}"
EVAL_SGLANG_SCRIPT="${EXP_DIR}/../_shared/scripts/evaluate_model_sglang.py"
EVAL_DATA_DIR="${EXP_DIR}/../_shared/eval_data"
EVAL_TEMPERATURE=0.6
EVAL_TOP_P=0.95
EVAL_MAX_CONCURRENT=256

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
        > /tmp/vllm_dskd_eval.log 2>&1 &
    local spid=$!

    for i in $(seq 1 60); do
        sleep 5
        if ! kill -0 $spid 2>/dev/null; then
            log "  Server died!"; tail -5 /tmp/vllm_dskd_eval.log; return 1
        fi
        if curl -s http://127.0.0.1:${EVAL_PORT}/v1/models 2>/dev/null | grep -q "data"; then
            log "  Server ready!"; return 0
        fi
    done
    log "  Timeout!"; return 1
}

# ==============================================================================
# Step 1: Clean old DSKD and retrain
# ==============================================================================
log "Step 1: Cleaning old DSKD directory..."
rm -rf "${DSKD_LOCAL_DIR}"

log "Step 2: Starting DSKD training (SPEED_TIER=safe)..."
DSKD_TRAIN_LOG="${LOG_DIR}/train_dskd_${TIMESTAMP}.log"
(
    cd "${DSKD_METHOD_DIR}" && \
    SPEED_TIER=safe bash launch.sh
) > "${DSKD_TRAIN_LOG}" 2>&1
DSKD_RC=$?

if [ ${DSKD_RC} -ne 0 ]; then
    log "DSKD training FAILED (rc=${DSKD_RC}). Check: ${DSKD_TRAIN_LOG}"
    log "Last 20 lines:"
    tail -20 "${DSKD_TRAIN_LOG}"
    exit 1
fi
log "Step 2: DSKD training completed successfully!"

# ==============================================================================
# Step 3: Merge FSDP checkpoints to HF
# ==============================================================================
log "Step 3: Merging FSDP checkpoints to HF..."
MERGE_SCRIPT="${EXP_DIR}/scripts/merge_fsdp_to_hf.py"
BASE_MODEL="/path/to/models/phi-4-mini-instruct"

shopt -s nullglob
FSDP_CKPTS=( "${DSKD_LOCAL_DIR}/fsdp"/global_step_* )
shopt -u nullglob

if [ ${#FSDP_CKPTS[@]} -eq 0 ]; then
    log "  No FSDP checkpoints found! Exiting."
    exit 1
fi

mkdir -p "${DSKD_LOCAL_DIR}/hf"
for fsdp_dir in "${FSDP_CKPTS[@]}"; do
    step_name=$(basename "${fsdp_dir}")
    hf_out="${DSKD_LOCAL_DIR}/hf/${step_name}"
    if [ -d "${hf_out}" ] && [ -f "${hf_out}/model.safetensors" -o -f "${hf_out}/model.safetensors.index.json" ]; then
        log "  [${step_name}] Already merged, skip."
        continue
    fi
    log "  [${step_name}] Merging..."
    ${PYTHON} ${MERGE_SCRIPT} \
        --fsdp_dir "${fsdp_dir}" \
        --hf_dir "${hf_out}" \
        --base_model "${BASE_MODEL}" \
        2>&1 | tail -3
    log "  [${step_name}] Merged."
done
log "Step 3: All checkpoints merged."

# ==============================================================================
# Step 4: Evaluate all checkpoints on all benchmarks
# ==============================================================================
log "Step 4: Evaluating DSKD checkpoints..."
mkdir -p "${RESULTS_DIR}"

shopt -s nullglob
HF_CKPTS=( "${DSKD_LOCAL_DIR}/hf"/global_step_* )
shopt -u nullglob

gpu_id=0
for merged_dir in "${HF_CKPTS[@]}"; do
    step_name=$(basename "${merged_dir}")
    tag="dskd_phi4mini_${step_name}"

    if [ ! -f "${merged_dir}/model.safetensors" ] && [ ! -f "${merged_dir}/model.safetensors.index.json" ]; then
        log "  [${step_name}] Not a valid HF checkpoint, skip."
        continue
    fi

    # Check if all benchmarks already done
    all_done=true
    for bench in "${ALL_BENCHMARKS[@]}"; do
        [ ! -f "${RESULTS_DIR}/${tag}_${bench}_details.json" ] && { all_done=false; break; }
    done
    if [ "${all_done}" = true ]; then
        log "  [${step_name}] All benchmarks done, skip."
        continue
    fi

    # Start vLLM server
    eval_kill_server
    if ! eval_start_server "${merged_dir}" "${gpu_id}"; then
        log "  [${step_name}] First attempt failed, retrying on next GPU..."
        eval_kill_server; sleep 5
        gpu_id=$(( (gpu_id + 1) % 8 ))
        if ! eval_start_server "${merged_dir}" "${gpu_id}"; then
            log "  [${step_name}] FAILED to start server, skip."
            eval_kill_server; continue
        fi
    fi

    # Evaluate all benchmarks
    for bench in "${ALL_BENCHMARKS[@]}"; do
        output_dir="${RESULTS_DIR}/${tag}_${bench}"
        details_json="${RESULTS_DIR}/${tag}_${bench}_details.json"

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

# ==============================================================================
# Final Summary
# ==============================================================================
log ""
log "=========================================="
log "=== DSKD ALL DONE ==="
log "=========================================="
log "Results: ${RESULTS_DIR}/"
log ""

# Print metrics summary
for f in "${RESULTS_DIR}"/*_details.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f" _details.json)
    score=$(${PYTHON} -c "import json; d=json.load(open('$f')); print(f\"{d.get('accuracy', d.get('pass_rate', 'N/A'))}\")" 2>/dev/null || echo "N/A")
    echo "  ${fname}: ${score}"
done
log "=========================================="
