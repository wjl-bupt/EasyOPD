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

# Auto-detect Python environment
# Override: USE_SJ_ENV=1 to force OpenAgentRL-sj, or set PYTHON=/path/to/python directly
if [ -n "${PYTHON:-}" ]; then
    # User explicitly set PYTHON
    _PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    SITE_PACKAGES="$(dirname $(dirname $PYTHON))/lib/python${_PY_VER}/site-packages"
elif [ "${USE_SJ_ENV:-0}" = "1" ] || [ ! -x "/opt/conda/envs/OpenAgentRL-sglang/bin/python" ]; then
    PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
    _PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    SITE_PACKAGES="/opt/conda/envs/OpenAgentRL-sj/lib/python${_PY_VER}/site-packages"
elif [ -x "/opt/conda/envs/OpenAgentRL-sglang/bin/python" ]; then
    PYTHON="/opt/conda/envs/OpenAgentRL-sglang/bin/python"
    SITE_PACKAGES="/opt/conda/envs/OpenAgentRL-sglang/lib/python3.11/site-packages"
else
    echo "ERROR: No suitable Python environment found (tried OpenAgentRL-sglang, OpenAgentRL-sj)"
    exit 1
fi
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cu13/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
echo "Using Python: ${PYTHON}"
echo "Site-packages: ${SITE_PACKAGES}"
EVAL_SCRIPT="/path/to/EasyOPD/experiments/_shared/scripts/evaluate_model_sglang.py"
DATA_DIR="/path/to/EasyOPD/experiments/_shared/eval_data"
METHODS_DIR="/path/to/EasyOPD/experiments/01_cross_tokenizer_opd/methods"

PORT=30000
BASE_URL="http://127.0.0.1:${PORT}"
DP_SIZE=8
TP_SIZE=1
MAX_CONCURRENT=256

# Per-benchmark max_new_tokens (aligned with KDFlow paper config)
# Format: benchmark_name -> max_tokens
declare -A BENCH_MAX_TOKENS
BENCH_MAX_TOKENS[gsm8k]=4096
BENCH_MAX_TOKENS[math500]=4096
BENCH_MAX_TOKENS[mbpp]=2048
BENCH_MAX_TOKENS[live-code-bench-v6]=4096

# Sampling parameters (aligned with KDFlow paper config: t=0.6, top_p=0.95).
# Greedy decoding (t=0) causes phi-4-mini to enter repetition loops on MATH500
# (~40% of samples loop indefinitely), which dramatically lowers the score.
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}

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
    # # SFT
    # "sft:sft_phi4mini_global_step_78:/path/to/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini/hf/global_step_78"
    # "sft:sft_phi4mini_global_step_156:/path/to/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini/hf/global_step_156"

    # # Simple
    # "simple:simple_phi4mini_global_step_77:/path/to/models/runs/01_cross_tokenizer_opd/simple/simple_phi4mini/hf/global_step_77"
    # "simple:simple_phi4mini_global_step_154:/path/to/models/runs/01_cross_tokenizer_opd/simple/simple_phi4mini/hf/global_step_154"
    # "simple:simple_phi4mini_global_step_231:/path/to/models/runs/01_cross_tokenizer_opd/simple/simple_phi4mini/hf/global_step_231"
    # "simple:simple_phi4mini_global_step_308:/path/to/models/runs/01_cross_tokenizer_opd/simple/simple_phi4mini/hf/global_step_308"

    # # SimCT
    # "simct:simct_phi4mini_global_step_77:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf/global_step_77"
    # "simct:simct_phi4mini_global_step_154:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf/global_step_154"
    # "simct:simct_phi4mini_global_step_231:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf/global_step_231"
    # "simct:simct_phi4mini_global_step_308:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf/global_step_308"

    # # SimCT v2
    # "simct:simct_phi4mini_v2_global_step_77:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v2/hf/global_step_77"
    # "simct:simct_phi4mini_v2_global_step_154:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v2/hf/global_step_154"
    # "simct:simct_phi4mini_v2_global_step_231:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v2/hf/global_step_231"
    # "simct:simct_phi4mini_v2_global_step_308:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v2/hf/global_step_308"

    # # SimCT v3
    # "simct:simct_phi4mini_v3_global_step_77:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v3/hf/global_step_77"
    # "simct:simct_phi4mini_v3_global_step_154:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v3/hf/global_step_154"
    # "simct:simct_phi4mini_v3_global_step_231:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v3/hf/global_step_231"
    # "simct:simct_phi4mini_v3_global_step_308:/path/to/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini_v3/hf/global_step_308"

    # ALM
    # "alm:alm_phi4mini_global_step_77:/path/to/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_77"
    # "alm:alm_phi4mini_global_step_154:/path/to/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_154"
    # "alm:alm_phi4mini_global_step_231:/path/to/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_231"
    # "alm:alm_phi4mini_global_step_308:/path/to/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_308"

    # ULD
    # "uld:uld_phi4mini_global_step_77:/path/to/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_77"
    # "uld:uld_phi4mini_global_step_154:/path/to/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_154"
    # "uld:uld_phi4mini_global_step_231:/path/to/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_231"
    # "uld:uld_phi4mini_global_step_308:/path/to/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_308"

    # DSKD
    # "dskd:dskd_phi4mini_global_step_77:/path/to/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_77"
    # "dskd:dskd_phi4mini_global_step_154:/path/to/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_154"
    # "dskd:dskd_phi4mini_global_step_231:/path/to/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_231"
    # "dskd:dskd_phi4mini_global_step_308:/path/to/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_308"

    # Base models (control group)
    "base:base_qwen25_7b:/path/to/models/Qwen2.5-7B-Instruct"
    "base:base_phi4mini:/path/to/models/Phi-4-mini-instruct"
)

# Dynamically discover SFT checkpoints from hf/ directory
_SFT_HF_DIR="/path/to/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini/hf"
if [ -d "${_SFT_HF_DIR}" ]; then
    for _ckpt_dir in $(find "${_SFT_HF_DIR}" -maxdepth 1 -name "global_step_*" -type d | sort -t_ -k3 -n); do
        _step_name=$(basename "${_ckpt_dir}")
        MODELS+=("sft:sft_phi4mini_${_step_name}:${_ckpt_dir}")
    done
fi

# ======================== Helper Functions ========================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_server() {
    local url="${BASE_URL}/health"
    local max_wait=600
    local waited=0
    local interval=10

    while [ $waited -lt $max_wait ]; do
        # Check if server process is still alive
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ""
            log "  ✗ Server process died!"
            log "  Last 30 lines of log:"
            tail -30 /tmp/sglang_server.log
            return 1
        fi
        # Check if child process crashed (early exit instead of waiting 600s)
        if grep -q "Child process unexpectedly failed\|Scheduler hit an exception\|ModuleNotFoundError\|ImportError" /tmp/sglang_server.log 2>/dev/null; then
            echo ""
            log "  ✗ Server child process crashed!"
            log "  Last 30 lines of log:"
            tail -30 /tmp/sglang_server.log
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
        # Show last meaningful log line for progress
        local last_line
        last_line=$(tail -1 /tmp/sglang_server.log 2>/dev/null | head -c 120)
        echo -ne "\r  Waiting for server... ${waited}s / ${max_wait}s | ${last_line}    "
    done
    echo ""
    log "  Last 30 lines of log:"
    tail -30 /tmp/sglang_server.log
    return 1
}

kill_server() {
    # Kill all SGLang/vLLM processes on the target port
    local pids
    pids=$(ps aux | grep -E "[s]glang.launch_server|[v]llm.entrypoints.openai" | grep "\-\-port ${PORT}" | awk '{print $2}' || true)
    if [ -n "$pids" ]; then
        log "Killing existing server(s) on port ${PORT} (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    # Also kill any sglang/vllm child processes
    local child_pids
    child_pids=$(ps aux | grep -E "sglang::(scheduler|detokenizer|data_parallel)|vllm.v1.engine" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$child_pids" ]; then
        log "Killing child processes..."
        echo "$child_pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    # Final check: try curl to see if port is still responding
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q "200"; then
        log "WARNING: Port ${PORT} still responding after kill attempt, trying harder..."
        pkill -9 -f "sglang.launch_server\|vllm.entrypoints" 2>/dev/null || true
        sleep 5
    fi
}

start_server() {
    local model_path="$1"

    # ===== VLLM Backend =====
    if [ "${USE_VLLM:-0}" = "1" ]; then
        log "Starting vLLM server: TP=${TP_SIZE}, port=${PORT}"
        log "  Model: ${model_path}"

        # Use CUDA_VISIBLE_DEVICES to select GPU (default: 0)
        local vllm_gpu="${VLLM_GPU:-0}"
        CUDA_VISIBLE_DEVICES=${vllm_gpu} ${PYTHON} -m vllm.entrypoints.openai.api_server \
            --model "${model_path}" \
            --tensor-parallel-size ${TP_SIZE} \
            --port ${PORT} \
            --trust-remote-code \
            --gpu-memory-utilization 0.90 \
            --seed 42 \
            --max-model-len 16384 \
            --disable-log-requests \
            --enforce-eager \
            > /tmp/sglang_server.log 2>&1 &

        SERVER_PID=$!
        log "  Server PID: ${SERVER_PID}"

        echo -n "  "
        if wait_for_server; then
            echo ""
            log "  ✓ vLLM Server ready!"
            return 0
        else
            echo ""
            log "  ✗ vLLM Server failed to start within timeout!"
            log "  Check /tmp/sglang_server.log for details"
            kill_server
            return 1
        fi
    fi

    # ===== SGLang Backend (default) =====
    log "Starting SGLang server: DP=${DP_SIZE}, TP=${TP_SIZE}, port=${PORT}"
    log "  Model: ${model_path}"

    # Validate environment: check critical libraries exist
    local missing_libs=0
    if [ ! -f "${SITE_PACKAGES}/tvm_ffi/lib/libtvm_ffi.so" ]; then
        log "  WARNING: libtvm_ffi.so not found in ${SITE_PACKAGES}/tvm_ffi/lib/"
        missing_libs=1
    fi
    if ! find "${SITE_PACKAGES}" -name "libcudart.so.1*" -print -quit 2>/dev/null | grep -q .; then
        log "  WARNING: No libcudart.so found in site-packages. deep_gemm may fail."
        log "  Try: pip install cuda-toolkit==13.0.2"
        missing_libs=1
    fi

    # Fix config.json: rename rope_parameters -> rope_scaling for transformers 4.46.3 compat
    # (verl uses transformers 5.3.0 which renames rope_scaling -> rope_parameters;
    #  SGLang uses transformers 4.46.3 which only reads rope_scaling.
    #  Without this fix, LongRoPE config is lost and model performance drops ~8%!)
    local model_config="${model_path}/config.json"
    if [ -f "${model_config}" ] && grep -q '"rope_parameters"' "${model_config}" && ! grep -q '"rope_scaling"' "${model_config}"; then
        log "  Fixing config.json (rope_parameters -> rope_scaling for transformers 4.46.3 compat)..."
        ${PYTHON} -c "
import json
with open('${model_config}') as f:
    config = json.load(f)
if 'rope_parameters' in config and 'rope_scaling' not in config:
    config['rope_scaling'] = config.pop('rope_parameters')
    # Remove extra fields not recognized by older transformers
    if isinstance(config['rope_scaling'], dict):
        config['rope_scaling'].pop('original_max_position_embeddings', None)
        config['rope_scaling'].pop('partial_rotary_factor', None)
        config['rope_scaling'].pop('rope_theta', None)
        config['rope_scaling'].pop('rope_type', None)
    with open('${model_config}', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print('  Done: rope_parameters -> rope_scaling')
"
    fi

    # Fix tokenizer_config.json if it contains unsupported "TokenizersBackend" class
    # (saved by newer transformers>=4.50, not recognized by transformers==4.46.3)
    local tok_config="${model_path}/tokenizer_config.json"
    if [ -f "${tok_config}" ] && grep -q '"TokenizersBackend"' "${tok_config}"; then
        log "  Fixing tokenizer_config.json (TokenizersBackend -> GPT2Tokenizer)..."
        sed -i 's/"TokenizersBackend"/"GPT2Tokenizer"/g' "${tok_config}"
    fi

    # Inject chat_template from chat_template.jinja into tokenizer_config.json
    # (SGLang 0.4.6 + transformers 4.46.3 don't read chat_template.jinja automatically;
    #  this is needed for /v1/chat/completions to work correctly)
    local jinja_file="${model_path}/chat_template.jinja"
    if [ -f "${tok_config}" ] && [ -f "${jinja_file}" ] && ! grep -q '"chat_template"' "${tok_config}"; then
        log "  Injecting chat_template from chat_template.jinja into tokenizer_config.json..."
        ${PYTHON} -c "
import json, sys
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
        --random-seed 42 \
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

BACKEND_NAME="SGLang"
if [ "${USE_VLLM:-0}" = "1" ]; then
    BACKEND_NAME="vLLM"
fi
log "===== EasyOPD Model Evaluation (${BACKEND_NAME}) ====="
log "Methods: sft, simple, simct"
log "Benchmarks: ${BENCHMARKS_ARRAY[*]}"
log "Models: ${#MODELS[@]}"
log "DP=${DP_SIZE}, TP=${TP_SIZE}, Port=${PORT}"
log "Sampling: temperature=${TEMPERATURE}, top_p=${TOP_P} (KDFlow-aligned)"
echo ""

# Check data availability
log "Checking data availability..."
declare -A DATA_FILE_MAP
DATA_FILE_MAP[gsm8k]="gsm8k"
DATA_FILE_MAP[math500]="math500"
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

        # Use per-benchmark max_new_tokens (aligned with KDFlow)
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
