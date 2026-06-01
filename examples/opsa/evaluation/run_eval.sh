#!/bin/bash
# OPSA Safety Evaluation Script
# Usage: MODEL_PATH=<path> bash examples/opsa/evaluation/run_eval.sh

set -e

MODEL_PATH="${MODEL_PATH:-outputs/opsa/checkpoint}"
DATASETS="${DATASETS:-harmbench xstest wildjailbreak strongreject wildbenign}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GUARD_MODEL="${GUARD_MODEL:-wildguard}"
NUM_RUNS="${NUM_RUNS:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"

echo "============================================"
echo "OPSA Safety Evaluation"
echo "============================================"
echo "Model:    ${MODEL_PATH}"
echo "Datasets: ${DATASETS}"
echo "Output:   ${OUTPUT_DIR}"
echo "Guard:    ${GUARD_MODEL}"
echo "TP Size:  ${TP_SIZE}"
echo "============================================"

python examples/opsa/evaluation/run_safety_eval.py \
    --model_path "${MODEL_PATH}" \
    --datasets ${DATASETS} \
    --output_dir "${OUTPUT_DIR}" \
    --tp_size ${TP_SIZE} \
    --gpu_memory_utilization ${GPU_MEM} \
    --max_tokens ${MAX_TOKENS} \
    --guard_model "${GUARD_MODEL}" \
    --num_runs ${NUM_RUNS} \
    --temperature ${TEMPERATURE}

echo "============================================"
echo "Evaluation complete! Results in: ${OUTPUT_DIR}"
echo "============================================"
