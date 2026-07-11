#!/bin/bash
set -euo pipefail

# ==============================================================================
# Fill missing evaluation results for ULD and ALM
# ==============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/path/to/workspace/eval_logs"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

EXP_DIR="/path/to/EasyOPD/experiments/01_cross_tokenizer_opd"
RUNS_ROOT="/path/to/models/runs"
EXP_NAME="01_cross_tokenizer_opd"

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
        > /tmp/vllm_fill_eval.log 2>&1 &
    local spid=$!

    for i in $(seq 1 60); do
        sleep 5
        if ! kill -0 $spid 2>/dev/null; then
            log "  Server died!"; tail -5 /tmp/vllm_fill_eval.log; return 1
        fi
        if curl -s http://127.0.0.1:${EVAL_PORT}/v1/models 2>/dev/null | grep -q "data"; then
            log "  Server ready!"; return 0
        fi
    done
    log "  Timeout!"; return 1
}

eval_one_model() {
    local method="$1" step="$2" gpu_id="$3"
    shift 3
    local benchmarks=("$@")

    local model_path="${RUNS_ROOT}/${EXP_NAME}/${method}/${method}_phi4mini/hf/${step}"
    local results_dir="${EXP_DIR}/methods/${method}/results/${method}_phi4mini"
    local tag="${method}_phi4mini_${step}"

    if [ ! -d "${model_path}" ]; then
        log "  [${method}/${step}] Checkpoint not found: ${model_path}, SKIP"
        return 1
    fi

    # Clean empty prediction files from previous failed attempts
    for bench in "${benchmarks[@]}"; do
        local output_dir="${results_dir}/${tag}_${bench}"
        if [ -d "${output_dir}" ]; then
            local pred_file="${output_dir}/predictions.jsonl"
            if [ -f "${pred_file}" ] && [ ! -s "${pred_file}" ]; then
                log "  [${method}/${step}] Removing empty predictions for ${bench}"
                rm -rf "${output_dir}"
            fi
        fi
    done

    # Start vLLM server
    eval_kill_server
    if ! eval_start_server "${model_path}" "${gpu_id}"; then
        log "  [${method}/${step}] First attempt failed, retrying on next GPU..."
        eval_kill_server; sleep 5
        gpu_id=$(( (gpu_id + 1) % 8 ))
        if ! eval_start_server "${model_path}" "${gpu_id}"; then
            log "  [${method}/${step}] FAILED to start server, skip."
            eval_kill_server; return 1
        fi
    fi

    # Evaluate each benchmark
    for bench in "${benchmarks[@]}"; do
        local output_dir="${results_dir}/${tag}_${bench}"
        local details_json="${results_dir}/${tag}_${bench}_details.json"

        if [ -f "${details_json}" ]; then
            log "  [${method}/${step}] ${bench}: already done, skip."
            continue
        fi

        mkdir -p "${output_dir}"
        log "  [${method}/${step}] Evaluating ${bench}..."
        ${PYTHON} ${EVAL_SGLANG_SCRIPT} --model_path "${model_path}" --dataset "${bench}" \
            --base_url "${EVAL_BASE_URL}" --output_dir "${output_dir}" \
            --data_dir "${EVAL_DATA_DIR}" \
            --max_new_tokens ${BENCH_TOKENS[$bench]} --max_concurrent ${EVAL_MAX_CONCURRENT} \
            --temperature ${EVAL_TEMPERATURE} --top_p ${EVAL_TOP_P} \
            2>&1 | tail -5
        if [ $? -eq 0 ] && [ -f "${output_dir}/metrics.json" ]; then
            cp "${output_dir}/metrics.json" "${details_json}"
            log "  [${method}/${step}] ${bench}: Done"
        else
            log "  [${method}/${step}] ${bench}: Failed"
        fi
    done

    eval_kill_server; sleep 5
    return 0
}

# ==============================================================================
# Main: Fill missing evaluations
# ==============================================================================
log "=========================================="
log "=== Filling Missing Evaluations ==="
log "=========================================="

gpu_id=0

# 1. ULD step_77: math500, gsm8k, mbpp
log ""
log "=== ULD global_step_77: math500, gsm8k, mbpp ==="
eval_one_model "uld" "global_step_77" ${gpu_id} "math500" "gsm8k" "mbpp"
gpu_id=$(( (gpu_id + 1) % 8 ))

# 2. ALM step_77: math500, gsm8k, mbpp
log ""
log "=== ALM global_step_77: math500, gsm8k, mbpp ==="
eval_one_model "alm" "global_step_77" ${gpu_id} "math500" "gsm8k" "mbpp"
gpu_id=$(( (gpu_id + 1) % 8 ))

# 3. ALM step_308: live-code-bench-v6
log ""
log "=== ALM global_step_308: live-code-bench-v6 ==="
eval_one_model "alm" "global_step_308" ${gpu_id} "live-code-bench-v6"

# ==============================================================================
# Final Summary
# ==============================================================================
log ""
log "=========================================="
log "=== Fill Missing Eval DONE ==="
log "=========================================="

log ""
log "--- ULD Results ---"
for f in "${EXP_DIR}/methods/uld/results/uld_phi4mini"/*_details.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f" _details.json)
    score=$(${PYTHON} -c "import json; d=json.load(open('$f')); print(d.get('score', 'N/A'))" 2>/dev/null || echo "N/A")
    echo "  ${fname}: ${score}"
done

log ""
log "--- ALM Results ---"
for f in "${EXP_DIR}/methods/alm/results/alm_phi4mini"/*_details.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f" _details.json)
    score=$(${PYTHON} -c "import json; d=json.load(open('$f')); print(d.get('score', 'N/A'))" 2>/dev/null || echo "N/A")
    echo "  ${fname}: ${score}"
done

log "=========================================="
