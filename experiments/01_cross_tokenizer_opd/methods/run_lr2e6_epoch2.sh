#!/usr/bin/env bash
# =============================================================================
# run_lr2e6_epoch2.sh
# One-shot script: clean old checkpoints + results, then retrain Simple & SimCT
# with ACTOR_LR=2e-6, TOTAL_EPOCHS=2.
#
# Usage:
#   cd /apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/01_cross_tokenizer_opd/methods
#   bash run_lr2e6_epoch2.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="/root/workspace/models/runs/01_cross_tokenizer_opd"

# Directories to DELETE (old training artifacts from lr=5e-7, epochs=1 run)
SIMPLE_FSDP="${RUNS_ROOT}/simple/simple_phi4mini/fsdp"
SIMPLE_HF="${RUNS_ROOT}/simple/simple_phi4mini/hf"
SIMPLE_LOGS="${RUNS_ROOT}/simple/simple_phi4mini/logs"
SIMCT_FSDP="${RUNS_ROOT}/simct/simct_phi4mini/fsdp"
SIMCT_HF="${RUNS_ROOT}/simct/simct_phi4mini/hf"
SIMCT_LOGS="${RUNS_ROOT}/simct/simct_phi4mini/logs"
SIMPLE_RESULTS="${SCRIPT_DIR}/simple/results"
SIMCT_RESULTS="${SCRIPT_DIR}/simct/results"

# Directory to PRESERVE (SFT warmup checkpoint - must NOT be deleted!)
SFT_CKPT="${RUNS_ROOT}/sft/sft_phi4mini/hf/global_step_78"

echo "============================================================"
echo "⚠️  DESTRUCTIVE OPERATION WARNING ⚠️"
echo "============================================================"
echo ""
echo "This script will DELETE the following directories:"
echo ""
echo "  [Simple FSDP ckpts]  ${SIMPLE_FSDP}"
echo "  [Simple HF ckpts]    ${SIMPLE_HF}"
echo "  [Simple logs]        ${SIMPLE_LOGS}"
echo "  [SimCT FSDP ckpts]   ${SIMCT_FSDP}"
echo "  [SimCT HF ckpts]     ${SIMCT_HF}"
echo "  [SimCT logs]         ${SIMCT_LOGS}"
echo "  [Simple eval results] ${SIMPLE_RESULTS}"
echo "  [SimCT eval results]  ${SIMCT_RESULTS}"
echo ""
echo "⚠️  This is IRREVERSIBLE. Old checkpoints and eval results will be lost."
echo ""
echo "The following will be PRESERVED (SFT warmup):"
echo "  ✅ ${SFT_CKPT}"
echo ""
echo "New training config: ACTOR_LR=2e-6, TOTAL_EPOCHS=2 (308 steps)"
echo "============================================================"
echo ""
read -p "Type 'yes' to confirm and proceed: " CONFIRM

if [ "${CONFIRM}" != "yes" ]; then
    echo "Aborted by user."
    exit 1
fi

echo ""
echo "[$(date)] Starting cleanup..."

# Clean old checkpoints
for DIR in "${SIMPLE_FSDP}" "${SIMPLE_HF}" "${SIMPLE_LOGS}" \
           "${SIMCT_FSDP}" "${SIMCT_HF}" "${SIMCT_LOGS}"; do
    if [ -d "${DIR}" ]; then
        echo "  Removing: ${DIR}"
        rm -rf "${DIR}"
    else
        echo "  (not found, skip): ${DIR}"
    fi
done

# Clean old evaluation results (files inside, keep directory)
for DIR in "${SIMPLE_RESULTS}" "${SIMCT_RESULTS}"; do
    if [ -d "${DIR}" ]; then
        echo "  Cleaning results in: ${DIR}"
        rm -f "${DIR}"/*.json "${DIR}"/*.csv "${DIR}"/*.txt 2>/dev/null || true
    else
        echo "  (not found, skip): ${DIR}"
    fi
done

echo "[$(date)] Cleanup done."
echo ""

# Verify SFT checkpoint is intact
if [ ! -d "${SFT_CKPT}" ]; then
    echo "[FATAL] SFT warmup checkpoint missing: ${SFT_CKPT}"
    echo "Cannot proceed without the base model. Aborting."
    exit 1
fi
echo "✅ SFT warmup checkpoint verified: ${SFT_CKPT}"
echo ""

# ======================== Run Simple ========================
echo "============================================================"
echo "[$(date)] Starting Simple training (lr=2e-6, epochs=2)..."
echo "============================================================"
FORCE_RETRAIN=1 bash "${SCRIPT_DIR}/simple/launch.sh"
SIMPLE_EXIT=$?

if [ ${SIMPLE_EXIT} -ne 0 ]; then
    echo "[ERROR] Simple training failed with exit code ${SIMPLE_EXIT}"
    echo "Stopping. SimCT will NOT be started."
    exit ${SIMPLE_EXIT}
fi

echo ""
echo "[$(date)] Simple training completed successfully."
echo ""

# ======================== Run SimCT ========================
echo "============================================================"
echo "[$(date)] Starting SimCT training (lr=2e-6, epochs=2)..."
echo "============================================================"
FORCE_RETRAIN=1 bash "${SCRIPT_DIR}/simct/launch.sh"
SIMCT_EXIT=$?

if [ ${SIMCT_EXIT} -ne 0 ]; then
    echo "[ERROR] SimCT training failed with exit code ${SIMCT_EXIT}"
    exit ${SIMCT_EXIT}
fi

echo ""
echo "============================================================"
echo "[$(date)] ALL DONE! Both Simple and SimCT completed."
echo "============================================================"
echo ""
echo "Results:"
echo "  Simple: ${SIMPLE_RESULTS}/"
echo "  SimCT:  ${SIMCT_RESULTS}/"
