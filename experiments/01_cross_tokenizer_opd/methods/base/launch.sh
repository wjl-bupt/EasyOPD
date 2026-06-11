#!/bin/bash
set -euo pipefail

# ============================================================
# Base model evaluation for Cross-Tokenizer OPD
# Evaluates the UNTRAINED student & teacher models as the reference
# baseline for every other method (sft / dskd / uld / simct / ...).
#
# Pipeline:
#   1. Evaluate student base model (Phi-4-mini-instruct)
#   2. Evaluate teacher base model (Qwen2.5-7B-Instruct)
#
# No training, no checkpoints, no merging. Pure inference.
# ============================================================

export PYTHONPATH="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"

# Paths
EASYOPD_ROOT="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD"
EXPERIMENT_DIR="${EASYOPD_ROOT}/experiments"
EXP_DIR="${EXPERIMENT_DIR}/01_cross_tokenizer_opd"
SHARED_SCRIPTS="${EXPERIMENT_DIR}/_shared/scripts"

METHOD_DIR="${EXP_DIR}/methods/base"
RESULTS_DIR="${METHOD_DIR}/results"
LOG_DIR="${METHOD_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# Local-disk model paths (loaded much faster than network /apdcephfs).
STUDENT_MODEL="/root/workspace/models/Phi-4-mini-instruct"
TEACHER_MODEL="/root/workspace/models/Qwen2.5-7B-Instruct"

# Evaluation tags shown in result filenames.
STUDENT_TAG="base_phi4mini"
TEACHER_TAG="base_qwen25_7b_instruct"

N_GPUS=8
BENCHMARKS=("math500" "gsm8k")

# ============================================================
# Helpers (mirror the skip logic in sft/launch.sh Step 3)
# ============================================================
# Returns 0 (true) if results/<TAG>_<BENCH>_details.json exists AND
# its 'total' field is > 0 (so a previously-failed run with total=0
# will NOT be treated as already done).
eval_already_done() {
    local details_json="$1"
    [ -f "${details_json}" ] || return 1
    local total
    total=$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('total',0))" "${details_json}" 2>/dev/null || echo 0)
    [ "${total}" -gt 0 ]
}

run_eval_one() {
    # $1 = MODEL_PATH, $2 = MODEL_TAG, $3 = BENCH
    local model_path="$1" tag="$2" bench="$3"
    local details_json="${RESULTS_DIR}/${tag}_${bench}_details.json"

    if [ "${FORCE_REEVAL:-0}" != "1" ] && eval_already_done "${details_json}"; then
        echo "[$(date)] [${tag}] ${bench}: existing valid result at ${details_json}, skipping."
        return 0
    fi

    if [ ! -d "${model_path}" ]; then
        echo "[$(date)] [${tag}] ERROR: model dir not found: ${model_path}"
        echo "[$(date)] [${tag}] Make sure the base model has been copied to local disk first."
        return 1
    fi

    echo "[$(date)] [${tag}] Evaluating ${bench} on ${model_path}"
    ${PYTHON} ${SHARED_SCRIPTS}/evaluate_model.py \
        --model_path "${model_path}" \
        --model_name "${tag}" \
        --output_dir "${RESULTS_DIR}" \
        --tensor_parallel_size 1 \
        --dp_size ${N_GPUS} \
        --benchmarks "${bench}" \
        2>&1 | tee "${LOG_DIR}/eval_${tag}_${bench}.log"
}

# ============================================================
# Step 1: Evaluate student base model
# ============================================================
echo "[$(date)] ===== Step 1: Evaluating STUDENT base model (${STUDENT_TAG}) ====="
for BENCH in "${BENCHMARKS[@]}"; do
    run_eval_one "${STUDENT_MODEL}" "${STUDENT_TAG}" "${BENCH}"
done

# ============================================================
# Step 2: Evaluate teacher base model
# ============================================================
echo "[$(date)] ===== Step 2: Evaluating TEACHER base model (${TEACHER_TAG}) ====="
for BENCH in "${BENCHMARKS[@]}"; do
    run_eval_one "${TEACHER_MODEL}" "${TEACHER_TAG}" "${BENCH}"
done

echo "[$(date)] ===== All Done ====="
echo "Eval results under: ${RESULTS_DIR}"
echo "(Tip: re-evaluate everything with: FORCE_REEVAL=1 bash $0)"
