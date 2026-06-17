#!/bin/bash
# =============================================================================
# One-click Model Evaluation for EasyOPD (sft, simple, simct)
# Powered by SGLang DP inference engine
#
# Benchmarks: gsm8k, math500, mbpp, live-code-bench-v6
#
# Architecture:
#   1. For each model: start SGLang server (DP=8) → run all benchmarks → shutdown
#   2. Skips models/benchmarks that already have results
#   3. Results saved to each method's results/ directory
#
# Usage:
#   bash eval_model_sglang.sh
#
# To re-evaluate all (ignore existing results):
#   FORCE_REEVAL=1 bash eval_model_sglang.sh
#
# To evaluate only specific benchmarks:
#   BENCHMARKS="gsm8k,math500" bash eval_model_sglang.sh
# =============================================================================

set -o pipefail

# ======================== Configuration ========================

PYTHON="/opt/conda/envs/OpenAgentRL-sglang/bin/python"
export LD_LIBRARY_PATH="/opt/conda/envs/OpenAgentRL-sglang/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/envs/OpenAgentRL-sglang/lib/python3.11/site-packages/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
EVAL_SCRIPT="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/_shared/scripts/evaluate_model_sglang.py"
DATA_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/_shared/eval_data"
METHODS_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/01_cross_tokenizer_opd/methods"

PORT=30000
BASE_URL="http://127.0.0.1:${PORT}"
DP_SIZE=8
TP_SIZE=1
MAX_TOKENS=4096
MAX_CONCURRENT=256

# Benchmarks to evaluate (can be overridden via env var)
if [ -z "${BENCHMARKS}" ]; then
    BENCHMARKS_ARRAY=("gsm8k" "math500" "mbpp" "live-code-bench-v6")
else
    IFS=',' read -ra BENCHMARKS_ARRAY <<< "${BENCHMARKS}"
fi

# ======================== Model List ========================
# Format: "METHOD:TAG:MODEL_PATH"
# Active: sft, simple, simct
# Commented out: alm, uld, dskd

MODELS=(
    # SFT
    "sft:sft_phi4mini_global_step_78:/root/workspace/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini/hf/global_step_78"
    "sft:sft_phi4mini_global_step_156:/root/workspace/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini/hf/global_step_156"

    # Simple
    "simple:simple_phi4mini_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/simple/simple_phi4mini/hf/global_step_77"
    "simple:simple_phi4mini_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/simple/simple_phi4mini/hf/global_step_154"
    "simple:simple_phi4mini_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/simple/simple_phi4mini/hf/global_step_231"
    "simple:simple_phi4mini_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/simple/simple_phi4mini/hf/global_step_308"

    # SimCT
    "simct:simct_phi4mini_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf/global_step_77"
    "simct:simct_phi4mini_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf/global_step_154"
    "simct:simct_phi4mini_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf/global_step_231"
    "simct:simct_phi4mini_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf/global_step_308"

    # SimCT v2
    "simct:simct_phi4mini_v2_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v2/hf/global_step_77"
    "simct:simct_phi4mini_v2_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v2/hf/global_step_154"
    "simct:simct_phi4mini_v2_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v2/hf/global_step_231"
    "simct:simct_phi4mini_v2_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v2/hf/global_step_308"

    # SimCT v3
    "simct:simct_phi4mini_v3_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v3/hf/global_step_77"
    "simct:simct_phi4mini_v3_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v3/hf/global_step_154"
    "simct:simct_phi4mini_v3_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v3/hf/global_step_231"
    "simct:simct_phi4mini_v3_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v3/hf/global_step_308"

    # # ALM (commented out)
    # "alm:alm_phi4mini_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_77"
    # "alm:alm_phi4mini_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_154"
    # "alm:alm_phi4mini_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_231"
    # "alm:alm_phi4mini_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_308"

    # # ULD (commented out)
    # "uld:uld_phi4mini_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_77"
    # "uld:uld_phi4mini_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_154"
    # "uld:uld_phi4mini_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_231"
    # "uld:uld_phi4mini_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_308"

    # # DSKD (commented out)
    # "dskd:dskd_phi4mini_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_77"
    # "dskd:dskd_phi4mini_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_154"
    # "dskd:dskd_phi4mini_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_231"
    # "dskd:dskd_phi4mini_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_308"
)

# ======================== Helper Functions ========================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_server() {
    local url="${BASE_URL}/health"
    local max_wait=300
    local waited=0
    local interval=5

    while [ $waited -lt $max_wait ]; do
        # Check if server process is still alive
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ""
            log "  ✗ Server process died!"
            log "  Last 20 lines of log:"
            tail -20 /tmp/sglang_server.log
            return 1
        fi
        if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "200"; then
            return 0
        fi
        if curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/v1/models" 2>/dev/null | grep -q "200"; then
            return 0
        fi
        sleep $interval
        waited=$((waited + interval))
        echo -ne "\r  Waiting for server... ${waited}s / ${max_wait}s"
    done
    echo ""
    return 1
}

kill_server() {
    local pids
    pids=$(lsof -ti :${PORT} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "Killing existing server on port ${PORT} (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 3
    fi
}

start_server() {
    local model_path="$1"
    log "Starting SGLang server: DP=${DP_SIZE}, TP=${TP_SIZE}, port=${PORT}"
    log "  Model: ${model_path}"

    # Fix tokenizer_config.json if it contains unsupported "TokenizersBackend" class
    # (saved by newer transformers>=4.50, not recognized by transformers==4.46.3)
    local tok_config="${model_path}/tokenizer_config.json"
    if [ -f "${tok_config}" ] && grep -q '"TokenizersBackend"' "${tok_config}"; then
        log "  Fixing tokenizer_config.json (TokenizersBackend -> GPT2Tokenizer)..."
        sed -i 's/"TokenizersBackend"/"GPT2Tokenizer"/g' "${tok_config}"
    fi

    ${PYTHON} -m sglang.launch_server \
        --model-path "${model_path}" \
        --dp-size ${DP_SIZE} \
        --tp-size ${TP_SIZE} \
        --port ${PORT} \
        --trust-remote-code \
        --mem-fraction-static 0.85 \
        --disable-cuda-graph \
        --grammar-backend none \
        --chat-template chatml \
        > /tmp/sglang_server.log 2>&1 &

    SERVER_PID=$!
    log "  Server PID: ${SERVER_PID}"

    echo -n "  "
    if wait_for_server; then
        echo ""
        log "  ✓ Server ready!"
        return 0
    else
        echo ""
        log "  ✗ Server failed to start within timeout!"
        log "  Check /tmp/sglang_server.log for details"
        kill_server
        return 1
    fi
}

results_exist() {
    local output_dir="$1"
    local pred_file="${output_dir}/predictions.jsonl"
    if [ -f "${pred_file}" ] && [ -s "${pred_file}" ]; then
        return 0
    fi
    return 1
}

# ======================== Main ========================

log "===== EasyOPD Model Evaluation (SGLang) ====="
log "Methods: sft, simple, simct"
log "Benchmarks: ${BENCHMARKS_ARRAY[*]}"
log "Models: ${#MODELS[@]}"
log "DP=${DP_SIZE}, TP=${TP_SIZE}, Port=${PORT}"
echo ""

# Check data availability
log "Checking data availability..."
declare -A DATA_FILE_MAP
DATA_FILE_MAP[gsm8k]="gsm8k_eval.parquet"
DATA_FILE_MAP[math500]="math500_eval.parquet"
DATA_FILE_MAP[mbpp]="mbpp"
DATA_FILE_MAP[live-code-bench-v6]="live-code-bench-v6"

for bench in "${BENCHMARKS_ARRAY[@]}"; do
    data_file="${DATA_FILE_MAP[$bench]}"
    data_path="${DATA_DIR}/${data_file}"
    if [ ! -e "${data_path}" ]; then
        log "ERROR: Data not found: ${data_path}"
        exit 1
    fi
    log "  ✓ ${bench}: ${data_path}"
done
echo ""

# Track statistics
TOTAL=${#MODELS[@]}
CURRENT=0
SKIPPED=0
SUCCESS=0
FAILED=0

# Kill any existing server
kill_server

# Process each model
for entry in "${MODELS[@]}"; do
    IFS=':' read -r method tag model_path <<< "${entry}"
    CURRENT=$((CURRENT + 1))

    echo ""
    log "============================================================"
    log "[${CURRENT}/${TOTAL}] ${tag}"
    log "============================================================"

    # Check model exists
    if [ ! -d "${model_path}" ]; then
        log "  ERROR: model not found: ${model_path}"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Check if all benchmarks already done
    all_done=true
    for bench in "${BENCHMARKS_ARRAY[@]}"; do
        output_dir="${METHODS_DIR}/${method}/results/${tag}/${bench}"
        if ! results_exist "${output_dir}" || [ "${FORCE_REEVAL:-0}" = "1" ]; then
            all_done=false
            break
        fi
    done

    if [ "${all_done}" = true ]; then
        log "  All benchmarks done, skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Start SGLang server for this model
    if ! start_server "${model_path}"; then
        log "  FAILED to start server, skipping."
        FAILED=$((FAILED + 1))
        continue
    fi

    # Run each benchmark
    model_failed=false
    for bench in "${BENCHMARKS_ARRAY[@]}"; do
        output_dir="${METHODS_DIR}/${method}/results/${tag}/${bench}"

        if results_exist "${output_dir}" && [ "${FORCE_REEVAL:-0}" != "1" ]; then
            log "  [${bench}] Already done, skipping."
            continue
        fi

        mkdir -p "${output_dir}"
        log "  [${bench}] Evaluating..."

        if ${PYTHON} ${EVAL_SCRIPT} \
            --model_path "${model_path}" \
            --dataset "${bench}" \
            --base_url "${BASE_URL}" \
            --output_dir "${output_dir}" \
            --data_dir "${DATA_DIR}" \
            --max_new_tokens ${MAX_TOKENS} \
            --max_concurrent ${MAX_CONCURRENT} \
            2>&1 | tee "${output_dir}/eval.log"; then
            log "  [${bench}] ✓ Done"
        else
            log "  [${bench}] ✗ Failed"
            model_failed=true
        fi
    done

    # Shutdown server
    log "  Shutting down server..."
    kill_server

    if [ "${model_failed}" = true ]; then
        FAILED=$((FAILED + 1))
    else
        SUCCESS=$((SUCCESS + 1))
    fi
done

# ======================== Summary ========================

echo ""
log "============================================================"
log "EVALUATION COMPLETE"
log "============================================================"
log "  Total models: ${TOTAL}"
log "  Success: ${SUCCESS}"
log "  Skipped: ${SKIPPED}"
log "  Failed:  ${FAILED}"
echo ""
log "Results saved to: ${METHODS_DIR}/{sft,simple,simct}/results/"
echo ""

# Print summary table
log "Summary Table:"
echo "----------------------------------------------------------------------"
printf "%-40s %-8s %-8s %-8s %-8s\n" "Model" "GSM8K" "MATH500" "MBPP" "LCB-v6"
echo "----------------------------------------------------------------------"
for entry in "${MODELS[@]}"; do
    IFS=':' read -r method tag model_path <<< "${entry}"
    gsm8k_score="-"
    math500_score="-"
    mbpp_score="-"
    lcb_score="-"

    gsm8k_metrics="${METHODS_DIR}/${method}/results/${tag}/gsm8k/metrics.json"
    math500_metrics="${METHODS_DIR}/${method}/results/${tag}/math500/metrics.json"
    mbpp_metrics="${METHODS_DIR}/${method}/results/${tag}/mbpp/metrics.json"
    lcb_metrics="${METHODS_DIR}/${method}/results/${tag}/live-code-bench-v6/metrics.json"

    if [ -f "${gsm8k_metrics}" ]; then
        gsm8k_score=$(${PYTHON} -c "import json; d=json.load(open('${gsm8k_metrics}')); print(f\"{d['score']*100:.1f}%\")" 2>/dev/null || echo "err")
    fi
    if [ -f "${math500_metrics}" ]; then
        math500_score=$(${PYTHON} -c "import json; d=json.load(open('${math500_metrics}')); print(f\"{d['score']*100:.1f}%\")" 2>/dev/null || echo "err")
    fi
    if [ -f "${mbpp_metrics}" ]; then
        mbpp_score=$(${PYTHON} -c "import json; d=json.load(open('${mbpp_metrics}')); print(f\"{d['score']*100:.1f}%\")" 2>/dev/null || echo "err")
    fi
    if [ -f "${lcb_metrics}" ]; then
        lcb_score=$(${PYTHON} -c "import json; d=json.load(open('${lcb_metrics}')); print(f\"{d['score']*100:.1f}%\")" 2>/dev/null || echo "err")
    fi

    printf "%-40s %-8s %-8s %-8s %-8s\n" "${tag}" "${gsm8k_score}" "${math500_score}" "${mbpp_score}" "${lcb_score}"
done
echo "----------------------------------------------------------------------"
