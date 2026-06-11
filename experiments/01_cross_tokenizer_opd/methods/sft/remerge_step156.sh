#!/bin/bash
# ============================================================
# Re-merge BOTH sft_phi4mini/fsdp/global_step_{78,156} -> hf/global_step_{78,156}
#
# ⚠️ DESTRUCTIVE: this script will delete the existing hf/global_step_78/ and
#    hf/global_step_156/ directories (if any) and rebuild them from scratch
#    using the FSDP shards. This is irreversible.
#    The fsdp/ side is read-only and never touched.
#
# Safety:
#   - Validates BOTH fsdp sources first (8 shards + fsdp_config.json +
#     huggingface/config.json each). If anything is missing, aborts BEFORE
#     deleting anything in hf/.
#   - Shows exactly what's about to be deleted, with sizes.
# ============================================================
set -euo pipefail
trap 'rc=$?; echo "[FATAL] $0 exited with code $rc at line $LINENO (last cmd: $BASH_COMMAND)" >&2' ERR

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
SHARED_SCRIPTS="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/_shared/scripts"
MERGE_SCRIPT="${SHARED_SCRIPTS}/merge_fsdp_checkpoint.py"

RUN_DIR="/root/workspace/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini"
FSDP_ROOT="${RUN_DIR}/fsdp"
HF_ROOT="${RUN_DIR}/hf"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

STEPS=(78 156)

# ------------------------------------------------------------
# Step 0: sanity checks (NO deletion happens in this section)
# ------------------------------------------------------------
echo "[$(date)] === Step 0: pre-flight checks (read-only, no deletion yet) ==="

if [ ! -f "${MERGE_SCRIPT}" ]; then
    echo "[FATAL] merge script not found: ${MERGE_SCRIPT}" >&2
    exit 1
fi

PROBLEM=0
for step in "${STEPS[@]}"; do
    src="${FSDP_ROOT}/global_step_${step}"
    echo "  -- checking FSDP source: ${src}"
    if [ ! -d "${src}" ]; then
        echo "    [FATAL] missing dir" >&2
        PROBLEM=1
        continue
    fi
    n_shards=$(ls "${src}"/model_world_size_8_rank_*.pt 2>/dev/null | wc -l)
    if [ "${n_shards}" -ne 8 ]; then
        echo "    [FATAL] expected 8 rank shards, got ${n_shards}" >&2
        PROBLEM=1
    fi
    if [ ! -f "${src}/fsdp_config.json" ]; then
        echo "    [FATAL] missing fsdp_config.json" >&2
        PROBLEM=1
    fi
    if [ ! -f "${src}/huggingface/config.json" ]; then
        echo "    [FATAL] missing huggingface/config.json" >&2
        PROBLEM=1
    fi
    if [ "${PROBLEM}" -eq 0 ]; then
        size=$(du -sh "${src}" 2>/dev/null | awk '{print $1}')
        echo "    OK (size=${size}, shards=${n_shards})"
    fi
done

if [ "${PROBLEM}" -ne 0 ]; then
    echo "[FATAL] FSDP source(s) incomplete; aborting BEFORE deleting any HF dir." >&2
    exit 1
fi

echo ""
echo "[$(date)] FSDP sources OK. About to delete the following HF dir(s):"
for step in "${STEPS[@]}"; do
    dst="${HF_ROOT}/global_step_${step}"
    if [ -d "${dst}" ]; then
        sz=$(du -sh "${dst}" 2>/dev/null | awk '{print $1}')
        echo "  -- ${dst}  (size=${sz})"
        ls -la "${dst}" 2>/dev/null | sed 's/^/       /'
    else
        echo "  -- ${dst}  (does not exist; will create fresh)"
    fi
done
echo ""

# ------------------------------------------------------------
# Step 1+2: per-step delete & remerge
# ------------------------------------------------------------
for step in "${STEPS[@]}"; do
    src="${FSDP_ROOT}/global_step_${step}"
    dst="${HF_ROOT}/global_step_${step}"
    log_file="${LOG_DIR}/merge_global_step_${step}.log"

    echo ""
    echo "============================================================"
    echo "[$(date)] [step ${step}] === Step 1: removing ${dst} ==="
    rm -rf "${dst}"
    echo "[$(date)] [step ${step}] removed."

    echo "[$(date)] [step ${step}] === Step 2: re-merging FSDP -> HF ==="
    echo "  src: ${src}"
    echo "  dst: ${dst}"
    echo "  log: ${log_file}"

    ${PYTHON} "${MERGE_SCRIPT}" \
        --checkpoint_dir "${src}" \
        --output_dir "${dst}" \
        2>&1 | tee "${log_file}"

    echo "[$(date)] [step ${step}] === Step 3: verifying output ==="
    if [ ! -f "${dst}/model.safetensors" ]; then
        echo "[FATAL] [step ${step}] merge produced no model.safetensors in ${dst}" >&2
        exit 1
    fi
    if [ ! -f "${dst}/config.json" ]; then
        echo "[FATAL] [step ${step}] merge produced no config.json in ${dst}" >&2
        exit 1
    fi
    echo "[$(date)] [step ${step}] OK:"
    du -sh "${dst}"
    ls -la "${dst}"
done

echo ""
echo "============================================================"
echo "[$(date)] === ALL DONE ==="
echo "Final HF layout under ${HF_ROOT}:"
ls -la "${HF_ROOT}"
echo ""
echo "Per-step sizes:"
for step in "${STEPS[@]}"; do
    du -sh "${HF_ROOT}/global_step_${step}" 2>/dev/null || true
done
