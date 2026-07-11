#!/bin/bash
# ==============================================================================
# run_all_methods.sh — Sequential training of ULD, DSKD, ALM methods
#
# This script runs three distillation methods sequentially (ULD → DSKD → ALM),
# each using the same starting checkpoint (global_step_116) as SimCT.
# For each method: clean old checkpoints → train → merge FSDP→HF → evaluate.
#
# Usage:
#   nohup bash run_all_methods.sh > /path/to/workspace/eval_logs/run_all_methods_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# ==============================================================================

# Exit on undefined variables but NOT on errors (we handle errors per-method)
set -u

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/path/to/workspace/eval_logs"
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/run_all_methods_${TIMESTAMP}.log"

# Paths
EXP_DIR="/path/to/EasyOPD/experiments/01_cross_tokenizer_opd"
RUNS_ROOT="/path/to/models/runs"
EXP_NAME="01_cross_tokenizer_opd"

# Methods to train in order
METHODS=("uld" "dskd" "alm")
RUN_NAMES=("uld_phi4mini" "dskd_phi4mini" "alm_phi4mini")

# Network disk checkpoint dirs
NET_CKPT_DIRS=(
    "${EXP_DIR}/methods/uld/checkpoints"
    "${EXP_DIR}/methods/dskd/checkpoints"
    "${EXP_DIR}/methods/alm/checkpoints"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

log "=========================================="
log "Starting sequential training: ULD → DSKD → ALM"
log "SPEED_TIER=safe, starting checkpoint=global_step_116"
log "=========================================="

# Track results
declare -A RESULTS

for i in "${!METHODS[@]}"; do
    METHOD="${METHODS[$i]}"
    RUN_NAME="${RUN_NAMES[$i]}"
    NET_CKPT="${NET_CKPT_DIRS[$i]}"
    LOCAL_RUN_DIR="${RUNS_ROOT}/${EXP_NAME}/${METHOD}/${RUN_NAME}"
    METHOD_DIR="${EXP_DIR}/methods/${METHOD}"
    METHOD_LOG="${LOG_DIR}/train_${METHOD}_${TIMESTAMP}.log"

    log ""
    log "=========================================="
    log "=== [${METHOD^^}] Starting method ${METHOD} (${i+1}/${#METHODS[@]}) ==="
    log "=========================================="

    # ---- Step 1: Clean old checkpoints ----
    log "[${METHOD^^}] Step 1: Cleaning old checkpoints..."

    if [ -d "${LOCAL_RUN_DIR}" ]; then
        log "[${METHOD^^}]   Removing local: ${LOCAL_RUN_DIR} ($(du -sh "${LOCAL_RUN_DIR}" 2>/dev/null | awk '{print $1}'))"
        rm -rf "${LOCAL_RUN_DIR}"
    else
        log "[${METHOD^^}]   Local dir not found, nothing to clean: ${LOCAL_RUN_DIR}"
    fi

    if [ -d "${NET_CKPT}" ]; then
        log "[${METHOD^^}]   Removing network: ${NET_CKPT} ($(du -sh "${NET_CKPT}" 2>/dev/null | awk '{print $1}'))"
        rm -rf "${NET_CKPT}"
    else
        log "[${METHOD^^}]   Network dir not found, nothing to clean: ${NET_CKPT}"
    fi

    log "[${METHOD^^}] Step 1: Cleanup done."

    # ---- Step 2: Train ----
    log "[${METHOD^^}] Step 2: Starting training..."
    log "[${METHOD^^}]   Log file: ${METHOD_LOG}"

    # Run with SPEED_TIER=safe to avoid OOM
    (
        cd "${METHOD_DIR}" && \
        SPEED_TIER=safe bash launch.sh
    ) > "${METHOD_LOG}" 2>&1
    TRAIN_RC=$?

    if [ ${TRAIN_RC} -ne 0 ]; then
        log "[${METHOD^^}] ❌ Training FAILED with exit code ${TRAIN_RC}"
        log "[${METHOD^^}]   Check log: ${METHOD_LOG}"
        RESULTS["${METHOD}"]="FAILED (train, rc=${TRAIN_RC})"
        log "[${METHOD^^}] Continuing to next method..."
        continue
    fi

    log "[${METHOD^^}] ✅ Training completed successfully."
    RESULTS["${METHOD}"]="SUCCESS"

    # Note: launch.sh already includes FSDP→HF merge (Step 4) and evaluation (Step 5)
    # So if training succeeds, merge and eval are already done.

    log "[${METHOD^^}] All steps completed."
done

# ---- Summary ----
log ""
log "=========================================="
log "=== FINAL SUMMARY ==="
log "=========================================="
for METHOD in "${METHODS[@]}"; do
    STATUS="${RESULTS[${METHOD}]:-NOT_RUN}"
    log "  ${METHOD^^}: ${STATUS}"
done
log "=========================================="
log "All methods processed. Master log: ${MASTER_LOG}"
log "Individual logs:"
for METHOD in "${METHODS[@]}"; do
    log "  ${METHOD}: ${LOG_DIR}/train_${METHOD}_${TIMESTAMP}.log"
done
log "=========================================="
