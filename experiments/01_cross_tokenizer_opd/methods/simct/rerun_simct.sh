#!/bin/bash
# 强制从头重跑 simct 训练
# 1. 清空旧的 FSDP checkpoint（verl 不会 resume）
# 2. 清空旧的 HF merged checkpoint
# 3. 启动训练

set -euo pipefail

FSDP_DIR="/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/fsdp"
HF_DIR="/root/workspace/models/runs/01_cross_tokenizer_opd/simct/simct_phi4mini/hf"

echo "=== Will remove all checkpoints under ==="
echo "  FSDP: ${FSDP_DIR}/global_step_*"
echo "  HF:   ${HF_DIR}/global_step_*"
echo ""

# 清空旧 checkpoint
rm -rf "${FSDP_DIR}"/global_step_*
rm -rf "${HF_DIR}"/global_step_*
echo "[$(date)] Old checkpoints removed."

# 启动训练
export FORCE_RETRAIN=1
export FORCE_REEVAL=1

cd /apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/01_cross_tokenizer_opd/methods/simct
bash launch.sh