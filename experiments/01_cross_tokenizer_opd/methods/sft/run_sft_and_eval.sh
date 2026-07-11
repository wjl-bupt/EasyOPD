#!/bin/bash
set -euo pipefail

# ============================================================
# SFT Training + Evaluation (one-click)
# ============================================================
# 1. Run SFT training (train_sft.sh)
# 2. Evaluate all SFT checkpoints on gsm8k, math500, mbpp, live-code-bench-v6
#
# Uses vLLM backend for evaluation (sglang has Phi-4-mini LongRoPE bug)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "${SCRIPT_DIR}")"
SHARED_DIR="${EXP_DIR}/_shared"
LOG_DIR="/path/to/workspace/eval_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
FULL_LOG="${LOG_DIR}/sft_train_and_eval_${TIMESTAMP}.log"

echo "============================================================"
echo "[$(date)] SFT Training + Evaluation Pipeline"
echo "============================================================"
echo "Full log: ${FULL_LOG}"
echo ""

# ============================================================
# Phase 1: SFT Training
# ============================================================
echo "[$(date)] ===== Phase 1: SFT Training ====="
echo ""

if bash "${SCRIPT_DIR}/train_sft.sh" 2>&1 | tee -a "${FULL_LOG}"; then
    echo ""
    echo "[$(date)] ✓ SFT Training completed successfully!"
    echo ""
else
    echo ""
    echo "[$(date)] ✗ SFT Training FAILED! Aborting."
    exit 1
fi

# ============================================================
# Phase 2: Evaluation
# ============================================================
echo "[$(date)] ===== Phase 2: Evaluating SFT Checkpoints ====="
echo ""

# Use vLLM backend (sglang has Phi-4-mini LongRoPE repetition bug in sj env)
# Evaluate only SFT models on all 4 benchmarks
export USE_SJ_ENV=1
export USE_VLLM=1
export VLLM_GPU=0
export BENCHMARKS="gsm8k,math500,mbpp,live-code-bench-v6"

# The eval_model_sglang.sh script evaluates all models defined in its MODELS array.
# We need to filter to only SFT models.
# Use the MODELS_FILTER env var if supported, otherwise we'll use the eval_all.sh with METHODS=sft

if bash "${SHARED_DIR}/eval_model_sglang.sh" 2>&1 | tee -a "${FULL_LOG}"; then
    echo ""
    echo "[$(date)] ✓ Evaluation completed successfully!"
else
    echo ""
    echo "[$(date)] ✗ Evaluation had some failures (check log for details)"
fi

echo ""
echo "============================================================"
echo "[$(date)] PIPELINE COMPLETE"
echo "============================================================"
echo "Full log: ${FULL_LOG}"
echo "============================================================"
