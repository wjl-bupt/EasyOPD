#!/usr/bin/env bash
# Phase 0: Evaluate base model on MATH-500 and GSM8K
set -euo pipefail

export PYTHONPATH="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD:${PYTHONPATH:-}"
PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
EXPERIMENT_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/benchmark"

echo "[$(date)] Starting base model evaluation..."

CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${EXPERIMENT_DIR}/evaluate_model.py \
    --model_path /apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-1.5B-Instruct \
    --model_name base_qwen2.5-1.5b \
    --output_dir ${EXPERIMENT_DIR}/results \
    --benchmarks "math500,gsm8k" \
    --max_tokens 2048 \
    --tensor_parallel_size 1

echo "[$(date)] Base model evaluation completed!"
