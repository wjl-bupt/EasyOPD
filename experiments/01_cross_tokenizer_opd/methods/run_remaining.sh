#!/bin/bash
# ============================================================
# run_remaining.sh
# Wait for ALM to finish, then run ULD and DSKD sequentially.
# Each method's launch.sh handles its own Ray start/stop, merge, and eval.
# ============================================================
set -euo pipefail

METHODS_DIR="$(cd "$(dirname "$0")" && pwd)"
ALM_LOG="${METHODS_DIR}/alm/run.log"
ULD_LOG="${METHODS_DIR}/uld/run.log"
DSKD_LOG="${METHODS_DIR}/dskd/run.log"

echo "[$(date)] === run_remaining.sh started ==="
echo "[$(date)] Waiting for ALM training (PID of launch.sh) to finish..."

# Wait for ALM launch.sh to finish (check if any process is running alm/launch.sh)
while pgrep -f "alm/launch.sh" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] ALM process finished."

# Check if ALM succeeded (look for "Step 5" or "Training Completed" in log)
if grep -q "Step 5: Evaluating" "${ALM_LOG}" 2>/dev/null || grep -q "Step 3: Training Completed" "${ALM_LOG}" 2>/dev/null; then
    echo "[$(date)] ALM appears to have completed successfully."
else
    echo "[$(date)] WARNING: ALM may have failed. Check ${ALM_LOG}"
    echo "[$(date)] Continuing with ULD anyway..."
fi

# ============================================================
# Run ULD
# ============================================================
echo "[$(date)] === Starting ULD ==="
/opt/conda/envs/OpenAgentRL-sj/bin/ray stop --force 2>/dev/null || true
sleep 5

bash "${METHODS_DIR}/uld/launch.sh" > "${ULD_LOG}" 2>&1
ULD_RC=$?

if [ ${ULD_RC} -eq 0 ]; then
    echo "[$(date)] ULD completed successfully (exit code 0)."
else
    echo "[$(date)] WARNING: ULD exited with code ${ULD_RC}. Check ${ULD_LOG}"
fi

# ============================================================
# Run DSKD
# ============================================================
echo "[$(date)] === Starting DSKD ==="
/opt/conda/envs/OpenAgentRL-sj/bin/ray stop --force 2>/dev/null || true
sleep 5

bash "${METHODS_DIR}/dskd/launch.sh" > "${DSKD_LOG}" 2>&1
DSKD_RC=$?

if [ ${DSKD_RC} -eq 0 ]; then
    echo "[$(date)] DSKD completed successfully (exit code 0)."
else
    echo "[$(date)] WARNING: DSKD exited with code ${DSKD_RC}. Check ${DSKD_LOG}"
fi

echo "[$(date)] === All three methods finished ==="
echo "  ALM log:  ${ALM_LOG}"
echo "  ULD log:  ${ULD_LOG}"
echo "  DSKD log: ${DSKD_LOG}"
