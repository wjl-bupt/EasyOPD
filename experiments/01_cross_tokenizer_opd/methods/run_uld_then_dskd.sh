#!/bin/bash
# Sequential runner: ULD -> DSKD
# This ensures no Ray conflicts between the two methods.
# Usage: nohup bash run_uld_then_dskd.sh > run_uld_then_dskd.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=============================================="
echo "[$(date)] Starting sequential run: ULD -> DSKD"
echo "=============================================="

# ===== Phase 1: ULD =====
echo ""
echo "[$(date)] ===== Phase 1: ULD ====="
echo ""

bash "${SCRIPT_DIR}/uld/launch.sh"
ULD_RC=$?

if [ ${ULD_RC} -ne 0 ]; then
    echo "[$(date)] [ERROR] ULD launch.sh exited with code ${ULD_RC}. Aborting DSKD."
    exit ${ULD_RC}
fi

echo ""
echo "[$(date)] ===== ULD Completed Successfully ====="
echo ""

# Brief pause to let GPU memory fully release
sleep 10

# ===== Phase 2: DSKD =====
echo ""
echo "[$(date)] ===== Phase 2: DSKD ====="
echo ""

bash "${SCRIPT_DIR}/dskd/launch.sh"
DSKD_RC=$?

if [ ${DSKD_RC} -ne 0 ]; then
    echo "[$(date)] [ERROR] DSKD launch.sh exited with code ${DSKD_RC}."
    exit ${DSKD_RC}
fi

echo ""
echo "=============================================="
echo "[$(date)] All done! Both ULD and DSKD completed."
echo "=============================================="
echo "ULD results:  ${SCRIPT_DIR}/uld/results/"
echo "DSKD results: ${SCRIPT_DIR}/dskd/results/"
