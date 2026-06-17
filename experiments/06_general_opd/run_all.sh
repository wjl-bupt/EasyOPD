#!/bin/bash
set -euo pipefail

# ============================================================
# 06_general_opd: Run All Methods Sequentially
#
# This script runs all three methods in the 06_general_opd experiment:
#   1. GRPO (baseline, no distillation)
#   2. GKD (on-policy + generalized JSD)
#   3. G-OPD (reverse KL advantages + reward extrapolation)
#
# Each method is independent and can be run separately.
# If a method has already completed (checkpoints exist), it will be skipped.
#
# Usage:
#   bash run_all.sh                    # Run all methods sequentially
#   FORCE_RETRAIN=1 bash run_all.sh    # Force retraining even if checkpoints exist
#   FORCE_REEVAL=1 bash run_all.sh     # Force re-evaluation even if results exist
#   METHODS="grpo gkd" bash run_all.sh # Run only specific methods
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}"

# ---- Mutual exclusion: only one run_all.sh instance at a time ----
LOCKFILE="${EXP_DIR}/.run_all.lock"
exec 9>"${LOCKFILE}"
if ! flock -n 9; then
    echo "[$(date)] ERROR: Another run_all.sh instance is already running (lockfile: ${LOCKFILE})."
    echo "         If you are sure no other instance is running, remove the lock:"
    echo "           rm -f ${LOCKFILE}"
    exit 1
fi
# Lock acquired; will be released automatically when script exits

# Default: run all three methods
METHODS="${METHODS:-grpo gkd g_opd}"

echo "============================================================"
echo " 06_general_opd: Sequential Method Runner"
echo "============================================================"
echo ""
echo " Student: Qwen2.5-1.5B-Instruct"
echo " Teacher: Qwen2.5-7B-Instruct"
echo " Methods: ${METHODS}"
echo " Started: $(date)"
echo ""
echo "============================================================"

TOTAL_START=$(date +%s)
FAILED_METHODS=()
SUCCEEDED_METHODS=()

for METHOD in ${METHODS}; do
    LAUNCH_SCRIPT="${EXP_DIR}/methods/${METHOD}/launch.sh"

    if [ ! -f "${LAUNCH_SCRIPT}" ]; then
        echo ""
        echo "[$(date)] ERROR: launch.sh not found for method '${METHOD}' at ${LAUNCH_SCRIPT}"
        FAILED_METHODS+=("${METHOD}")
        continue
    fi

    echo ""
    echo "============================================================"
    echo " [$(date)] Starting method: ${METHOD}"
    echo "============================================================"
    echo ""

    METHOD_START=$(date +%s)

    if bash "${LAUNCH_SCRIPT}"; then
        METHOD_END=$(date +%s)
        METHOD_ELAPSED=$(( METHOD_END - METHOD_START ))
        echo ""
        echo "[$(date)] ✅ Method '${METHOD}' completed successfully in ${METHOD_ELAPSED}s ($(( METHOD_ELAPSED / 60 ))m)"
        SUCCEEDED_METHODS+=("${METHOD}")
    else
        METHOD_END=$(date +%s)
        METHOD_ELAPSED=$(( METHOD_END - METHOD_START ))
        echo ""
        echo "[$(date)] ❌ Method '${METHOD}' FAILED after ${METHOD_ELAPSED}s ($(( METHOD_ELAPSED / 60 ))m)"
        FAILED_METHODS+=("${METHOD}")
        # Cleanup Ray after failure to ensure clean state for next method
        echo "[$(date)] Cleaning up Ray after failure..."
        /opt/conda/envs/OpenAgentRL-sj/bin/ray stop --force 2>/dev/null || true
        sleep 3
        rm -rf /tmp/ray/session_* /tmp/ray/ray_current_cluster /tmp/ray/*.json 2>/dev/null || true
        pkill -9 -f "ray::" 2>/dev/null || true
        pkill -9 -f "gcs_server" 2>/dev/null || true
        pkill -9 -f "raylet" 2>/dev/null || true
        sleep 2
        echo "[$(date)] Continuing to next method..."
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

echo ""
echo "============================================================"
echo " 06_general_opd: All Methods Complete"
echo "============================================================"
echo ""
echo " Total time: ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m)"
echo ""
echo " Succeeded (${#SUCCEEDED_METHODS[@]}): ${SUCCEEDED_METHODS[*]:-none}"
echo " Failed    (${#FAILED_METHODS[@]}): ${FAILED_METHODS[*]:-none}"
echo ""
echo "============================================================"

if [ ${#FAILED_METHODS[@]} -gt 0 ]; then
    exit 1
fi
