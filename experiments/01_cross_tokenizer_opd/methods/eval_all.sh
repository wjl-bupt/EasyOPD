#!/bin/bash
# =============================================================================
# Unified Evaluation Script for ALL trained models
#
# Models evaluated:
#   - Base: Phi-4-mini-instruct, Qwen2.5-7B-Instruct
#   - SFT: step 78, 156
#   - Simple: step 77, 154, 231, 308
#   - SimCT v1: step 77, 154, 231, 308
#   - SimCT v2: step 77, 154, 231, 308
#   - SimCT v3: step 77, 154, 231, 308
#
# Benchmarks: gsm8k, math500, mbpp, live-code-bench-v6
#
# Usage:
#   bash eval_all.sh
#
# To re-evaluate all (ignore existing results):
#   FORCE_REEVAL=1 bash eval_all.sh
#
# To evaluate only specific benchmarks:
#   BENCHMARKS="gsm8k,math500" bash eval_all.sh
#
# To evaluate only specific methods (comma-separated):
#   METHODS="base,sft" bash eval_all.sh
# =============================================================================

set -o pipefail

# ======================== Configuration ========================

# Auto-detect Python environment
if [ -x "/opt/conda/envs/OpenAgentRL-sglang/bin/python" ]; then
    PYTHON="/opt/conda/envs/OpenAgentRL-sglang/bin/python"
    SITE_PACKAGES="/opt/conda/envs/OpenAgentRL-sglang/lib/python3.11/site-packages"
elif [ -x "/opt/conda/envs/OpenAgentRL-sj/bin/python" ]; then
    PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
    _PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    SITE_PACKAGES="/opt/conda/envs/OpenAgentRL-sj/lib/python${_PY_VER}/site-packages"
else
    echo "ERROR: No suitable Python environment found"
    exit 1
fi
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cu13/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
echo "Using Python: ${PYTHON}"

# Shared paths
EVAL_SCRIPT="/path/to/EasyOPD/experiments/_shared/scripts/evaluate_model_sglang.py"
DATA_DIR="/path/to/EasyOPD/experiments/_shared/eval_data"
METHODS_DIR="/path/to/EasyOPD/experiments/01_cross_tokenizer_opd/methods"

# Server config
PORT=30000
BASE_URL="http://127.0.0.1:${PORT}"
DP_SIZE=8
TP_SIZE=1
MAX_CONCURRENT=256

# Per-benchmark max_new_tokens (aligned with KDFlow paper config)
declare -A BENCH_MAX_TOKENS
BENCH_MAX_TOKENS[gsm8k]=4096
BENCH_MAX_TOKENS[math500]=4096
BENCH_MAX_TOKENS[mbpp]=2048
BENCH_MAX_TOKENS[live-code-bench-v6]=4096

# Sampling parameters (aligned with KDFlow: t=0.6, top_p=0.95)
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}

# Benchmarks to evaluate
if [ -z "${BENCHMARKS}" ]; then
    BENCHMARKS_ARRAY=("gsm8k" "math500" "mbpp" "live-code-bench-v6")
else
    IFS=',' read -ra BENCHMARKS_ARRAY <<< "${BENCHMARKS}"
fi

# Methods filter (optional)
if [ -n "${METHODS}" ]; then
    IFS=',' read -ra METHODS_FILTER <<< "${METHODS}"
else
    METHODS_FILTER=()
fi

# ======================== Model List ========================
# Format: "METHOD:TAG:MODEL_PATH"
# METHOD determines the results subdirectory
# TAG is the unique identifier for this model's results
# MODEL_PATH is the local path to the HF model

MODELS_BASE="/path/to/models"
MODELS_RUNS="/path/to/models/runs/01_cross_tokenizer_opd"

MODELS=(
    # Base models
    "base:base_phi4mini:${MODELS_BASE}/Phi-4-mini-instruct"
    "base:base_qwen25_7b:${MODELS_BASE}/Qwen2.5-7B-Instruct"

    # SFT (Phi-4-mini) - KDFlow-aligned: lr=2e-6, max_len=2048, save_steps=20
    "sft:sft_phi4mini_global_step_20:${MODELS_RUNS}/sft/sft_phi4mini/hf/global_step_20"
    "sft:sft_phi4mini_global_step_40:${MODELS_RUNS}/sft/sft_phi4mini/hf/global_step_40"
    "sft:sft_phi4mini_global_step_60:${MODELS_RUNS}/sft/sft_phi4mini/hf/global_step_60"
    "sft:sft_phi4mini_global_step_80:${MODELS_RUNS}/sft/sft_phi4mini/hf/global_step_80"
    "sft:sft_phi4mini_global_step_100:${MODELS_RUNS}/sft/sft_phi4mini/hf/global_step_100"
    "sft:sft_phi4mini_global_step_108:${MODELS_RUNS}/sft/sft_phi4mini/hf/global_step_108"

    # Simple (Phi-4-mini)
    "simple:simple_phi4mini_global_step_77:${MODELS_RUNS}/simple/simple_phi4mini/hf/global_step_77"
    "simple:simple_phi4mini_global_step_154:${MODELS_RUNS}/simple/simple_phi4mini/hf/global_step_154"
    "simple:simple_phi4mini_global_step_231:${MODELS_RUNS}/simple/simple_phi4mini/hf/global_step_231"
    "simple:simple_phi4mini_global_step_308:${MODELS_RUNS}/simple/simple_phi4mini/hf/global_step_308"

    # SimCT v1 (Phi-4-mini)
    "simct:simct_v1_phi4mini_global_step_77:${MODELS_RUNS}/simct/simct_phi4mini/hf/global_step_77"
    "simct:simct_v1_phi4mini_global_step_154:${MODELS_RUNS}/simct/simct_phi4mini/hf/global_step_154"
    "simct:simct_v1_phi4mini_global_step_231:${MODELS_RUNS}/simct/simct_phi4mini/hf/global_step_231"
    "simct:simct_v1_phi4mini_global_step_308:${MODELS_RUNS}/simct/simct_phi4mini/hf/global_step_308"

    # SimCT v2 (Phi-4-mini)
    "simct:simct_v2_phi4mini_global_step_77:${MODELS_RUNS}/simct/simct_phi4mini_v2/hf/global_step_77"
    "simct:simct_v2_phi4mini_global_step_154:${MODELS_RUNS}/simct/simct_phi4mini_v2/hf/global_step_154"
    "simct:simct_v2_phi4mini_global_step_231:${MODELS_RUNS}/simct/simct_phi4mini_v2/hf/global_step_231"
    "simct:simct_v2_phi4mini_global_step_308:${MODELS_RUNS}/simct/simct_phi4mini_v2/hf/global_step_308"

    # SimCT v3 (Phi-4-mini)
    "simct:simct_v3_phi4mini_global_step_77:${MODELS_RUNS}/simct/simct_phi4mini_v3/hf/global_step_77"
    "simct:simct_v3_phi4mini_global_step_154:${MODELS_RUNS}/simct/simct_phi4mini_v3/hf/global_step_154"
    "simct:simct_v3_phi4mini_global_step_231:${MODELS_RUNS}/simct/simct_phi4mini_v3/hf/global_step_231"
    "simct:simct_v3_phi4mini_global_step_308:${MODELS_RUNS}/simct/simct_phi4mini_v3/hf/global_step_308"
)

# ======================== Helper Functions ========================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

should_eval_method() {
    local method="$1"
    if [ ${#METHODS_FILTER[@]} -eq 0 ]; then
        return 0  # No filter, evaluate all
    fi
    for m in "${METHODS_FILTER[@]}"; do
        if [ "$m" = "$method" ]; then
            return 0
        fi
    done
    return 1
}

wait_for_server() {
    local url="${BASE_URL}/health"
    local max_wait=600
    local waited=0
    local interval=10

    while [ $waited -lt $max_wait ]; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ""
            log "  ✗ Server process died!"
            tail -30 /tmp/sglang_server_eval.log
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
        local last_line
        last_line=$(tail -1 /tmp/sglang_server_eval.log 2>/dev/null | head -c 120)
        echo -ne "\r  Waiting for server... ${waited}s / ${max_wait}s | ${last_line}    "
    done
    echo ""
    tail -30 /tmp/sglang_server_eval.log
    return 1
}

kill_server() {
    local pids
    pids=$(ps aux | grep "[s]glang.launch_server" | grep "\-\-port ${PORT}" | awk '{print $2}' || true)
    if [ -n "$pids" ]; then
        log "Killing existing SGLang server(s) on port ${PORT} (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    local child_pids
    child_pids=$(ps aux | grep -E "sglang::(scheduler|detokenizer|data_parallel)" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$child_pids" ]; then
        log "Killing SGLang child processes..."
        echo "$child_pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q "200"; then
        log "WARNING: Port ${PORT} still responding, trying harder..."
        pkill -9 -f "sglang" 2>/dev/null || true
        sleep 5
    fi
}

start_server() {
    local model_path="$1"
    log "Starting SGLang server: DP=${DP_SIZE}, TP=${TP_SIZE}, port=${PORT}"
    log "  Model: ${model_path}"

    # Fix tokenizer_config.json if it contains unsupported "TokenizersBackend" class
    local tok_config="${model_path}/tokenizer_config.json"
    if [ -f "${tok_config}" ] && grep -q '"TokenizersBackend"' "${tok_config}"; then
        log "  Fixing tokenizer_config.json (TokenizersBackend -> GPT2Tokenizer)..."
        sed -i 's/"TokenizersBackend"/"GPT2Tokenizer"/g' "${tok_config}"
    fi

    # Inject chat_template from chat_template.jinja into tokenizer_config.json
    local jinja_file="${model_path}/chat_template.jinja"
    if [ -f "${tok_config}" ] && [ -f "${jinja_file}" ] && ! grep -q '"chat_template"' "${tok_config}"; then
        log "  Injecting chat_template from chat_template.jinja into tokenizer_config.json..."
        ${PYTHON} -c "
import json
with open('${tok_config}') as f:
    config = json.load(f)
with open('${jinja_file}') as f:
    config['chat_template'] = f.read().strip()
with open('${tok_config}', 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print('  Done: chat_template injected.')
"
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
        > /tmp/sglang_server_eval.log 2>&1 &

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
        log "  Check /tmp/sglang_server_eval.log for details"
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

log "============================================================"
log "  UNIFIED MODEL EVALUATION"
log "============================================================"
log "Benchmarks: ${BENCHMARKS_ARRAY[*]}"
log "DP=${DP_SIZE}, TP=${TP_SIZE}, Port=${PORT}"
log "Sampling: temperature=${TEMPERATURE}, top_p=${TOP_P}"
if [ ${#METHODS_FILTER[@]} -gt 0 ]; then
    log "Methods filter: ${METHODS_FILTER[*]}"
fi
echo ""

# Check data availability
log "Checking data availability..."
for bench in "${BENCHMARKS_ARRAY[@]}"; do
    data_path="${DATA_DIR}/${bench}"
    if [ ! -e "${data_path}" ]; then
        log "ERROR: Data not found: ${data_path}"
        exit 1
    fi
    log "  ✓ ${bench}: ${data_path}"
done
echo ""

# Filter models by method
FILTERED_MODELS=()
for entry in "${MODELS[@]}"; do
    IFS=':' read -r method tag model_path <<< "${entry}"
    if should_eval_method "${method}"; then
        FILTERED_MODELS+=("${entry}")
    fi
done

TOTAL=${#FILTERED_MODELS[@]}
log "Models to evaluate: ${TOTAL}"
echo ""

# Track statistics
CURRENT=0
SKIPPED=0
SUCCESS=0
FAILED=0
MISSING=0

# Kill any existing server
kill_server

# Process each model
for entry in "${FILTERED_MODELS[@]}"; do
    IFS=':' read -r method tag model_path <<< "${entry}"
    CURRENT=$((CURRENT + 1))

    echo ""
    log "============================================================"
    log "[${CURRENT}/${TOTAL}] ${tag}"
    log "  Method: ${method} | Path: ${model_path}"
    log "============================================================"

    # Check model exists
    if [ ! -d "${model_path}" ]; then
        log "  ⚠ Model not found: ${model_path}"
        log "  Skipping..."
        MISSING=$((MISSING + 1))
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

        # Use per-benchmark max_new_tokens
        bench_tokens=${BENCH_MAX_TOKENS[$bench]:-4096}

        if ${PYTHON} ${EVAL_SCRIPT} \
            --model_path "${model_path}" \
            --dataset "${bench}" \
            --base_url "${BASE_URL}" \
            --output_dir "${output_dir}" \
            --data_dir "${DATA_DIR}" \
            --max_new_tokens ${bench_tokens} \
            --max_concurrent ${MAX_CONCURRENT} \
            --temperature ${TEMPERATURE} \
            --top_p ${TOP_P} \
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
log "  EVALUATION COMPLETE"
log "============================================================"
log "  Total:   ${TOTAL}"
log "  Success: ${SUCCESS}"
log "  Skipped: ${SKIPPED} (already done)"
log "  Missing: ${MISSING} (model not found)"
log "  Failed:  ${FAILED}"
echo ""

# Print summary table
log "Results Summary:"
echo "========================================================================================"
printf "%-40s %-10s %-10s %-10s %-10s\n" "Model" "GSM8K" "MATH500" "MBPP" "LCB-v6"
echo "========================================================================================"

for entry in "${FILTERED_MODELS[@]}"; do
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

    printf "%-40s %-10s %-10s %-10s %-10s\n" "${tag}" "${gsm8k_score}" "${math500_score}" "${mbpp_score}" "${lcb_score}"
done
echo "========================================================================================"

if [ ${FAILED} -gt 0 ]; then
    log "WARNING: ${FAILED} model(s) failed!"
    exit 1
fi

log "All evaluations completed successfully!"
