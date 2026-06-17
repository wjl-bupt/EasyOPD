#!/bin/bash
set -euo pipefail

# ============================================================
# Batch Code Evaluation for ALM, ULD, DSKD only (MBPP + LCB-v6)
#
# Evaluates all trained checkpoints of ALM, ULD, DSKD on code benchmarks.
# Uses vLLM with DP=8 for fast inference.
#
# Prerequisites:
#   1. Run prepare_code_data.py first to download MBPP and LCB datasets
#   2. Model checkpoints must be on local disk (/root/workspace/models)
#
# Usage:
#   bash eval_code_alm_uld_dskd.sh
# ============================================================

export PYTHONPATH="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
EVAL_SCRIPT="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/_shared/scripts/evaluate_code.py"
DATA_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/_shared/eval_data"
METHODS_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/01_cross_tokenizer_opd/methods"

N_GPUS=8
MAX_TOKENS=2048
BENCHMARKS="mbpp,lcb"  # mbpp and live-code-bench-v6

# ============================================================
# Define models to evaluate (ALM + ULD + DSKD only)
# Format: "METHOD_NAME:MODEL_TAG:MODEL_PATH"
# ============================================================
MODELS=(
    # ALM
    "alm:alm_phi4mini_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_77"
    "alm:alm_phi4mini_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_154"
    "alm:alm_phi4mini_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_231"
    "alm:alm_phi4mini_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/alm/alm_phi4mini/hf/global_step_308"

    # DSKD
    "dskd:dskd_phi4mini_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_77"
    "dskd:dskd_phi4mini_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_154"
    "dskd:dskd_phi4mini_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_231"
    "dskd:dskd_phi4mini_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/dskd/dskd_phi4mini/hf/global_step_308"

    # ULD
    "uld:uld_phi4mini_global_step_77:/root/workspace/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_77"
    "uld:uld_phi4mini_global_step_154:/root/workspace/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_154"
    "uld:uld_phi4mini_global_step_231:/root/workspace/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_231"
    "uld:uld_phi4mini_global_step_308:/root/workspace/models/runs/01_cross_tokenizer_opd/uld/uld_phi4mini/hf/global_step_308"
)

# ============================================================
# Helper: check if result already exists
# ============================================================
eval_already_done() {
    local details_json="$1"
    [ -f "${details_json}" ] || return 1
    local total
    total=$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('total',0))" "${details_json}" 2>/dev/null || echo 0)
    [ "${total}" -gt 0 ]
}

# ============================================================
# Step 0: Check data availability
# ============================================================
echo "[$(date)] ===== Checking data availability ====="

for bench in $(echo "${BENCHMARKS}" | tr ',' ' '); do
    if [ "${bench}" = "mbpp" ]; then
        data_path="${DATA_DIR}/mbpp"
    elif [ "${bench}" = "lcb" ]; then
        data_path="${DATA_DIR}/live-code-bench-v6"
    fi
    if [ ! -d "${data_path}" ]; then
        echo "ERROR: Data not found: ${data_path}"
        echo "Please run first:"
        echo "  ${PYTHON} /apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/_shared/scripts/prepare_code_data.py --benchmarks ${bench}"
        exit 1
    fi
    echo "  ✓ ${bench} data: ${data_path}"
done

# ============================================================
# Step 1: Evaluate all models
# ============================================================
TOTAL=${#MODELS[@]}
CURRENT=0
SKIPPED=0
SUCCESS=0
FAILED=0

echo ""
echo "[$(date)] ===== Starting code evaluation (${TOTAL} models × ${BENCHMARKS}) ====="
echo ""

for entry in "${MODELS[@]}"; do
    IFS=':' read -r method tag model_path <<< "${entry}"
    CURRENT=$((CURRENT + 1))

    RESULTS_DIR="${METHODS_DIR}/${method}/results"
    mkdir -p "${RESULTS_DIR}"

    # Check if model exists on local disk
    if [ ! -d "${model_path}" ]; then
        # Try network disk path
        net_path=$(echo "${model_path}" | sed 's|/root/workspace/models|/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models|')
        if [ -d "${net_path}" ]; then
            echo "[${CURRENT}/${TOTAL}] [${tag}] Model not on local disk, need to copy first."
            echo "  Source: ${net_path}"
            echo "  Target: ${model_path}"
            echo "  Skipping (copy model to local disk first)."
            SKIPPED=$((SKIPPED + 1))
            continue
        else
            echo "[${CURRENT}/${TOTAL}] [${tag}] ERROR: model not found: ${model_path}"
            FAILED=$((FAILED + 1))
            continue
        fi
    fi

    # Check if all benchmarks already done
    all_done=true
    for bench in $(echo "${BENCHMARKS}" | tr ',' ' '); do
        details_json="${RESULTS_DIR}/${tag}_${bench}_details.json"
        if ! eval_already_done "${details_json}"; then
            all_done=false
            break
        fi
    done

    if [ "${all_done}" = true ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
        echo "[${CURRENT}/${TOTAL}] [${tag}] All benchmarks done, skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[${CURRENT}/${TOTAL}] [${tag}] Evaluating on ${BENCHMARKS}..."
    if ${PYTHON} ${EVAL_SCRIPT} \
        --model_path "${model_path}" \
        --model_name "${tag}" \
        --output_dir "${RESULTS_DIR}" \
        --data_dir "${DATA_DIR}" \
        --benchmarks "${BENCHMARKS}" \
        --max_tokens ${MAX_TOKENS} \
        --dp_size ${N_GPUS} \
        2>&1 | tee "${RESULTS_DIR}/${tag}_code_eval.log"; then
        echo "[${CURRENT}/${TOTAL}] [${tag}] ✓ Done"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "[${CURRENT}/${TOTAL}] [${tag}] ✗ Failed"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "[$(date)] Code Evaluation Complete"
echo "============================================================"
echo "  Total:   ${TOTAL}"
echo "  Success: ${SUCCESS}"
echo "  Skipped: ${SKIPPED}"
echo "  Failed:  ${FAILED}"
echo ""
echo "Results are in each method's results/ directory."
echo "(Re-run with FORCE_REEVAL=1 to re-evaluate all)"