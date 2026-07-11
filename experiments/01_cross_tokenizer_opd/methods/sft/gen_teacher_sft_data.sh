#!/bin/bash
# ============================================================================
# Generate Teacher SFT Data (aligned with KDFlow)
# ============================================================================
# This script regenerates teacher responses using the same pipeline as KDFlow:
#   1. Start SGLang server for Qwen2.5-7B-Instruct
#   2. Generate 8 trajectories per question via /v1/chat/completions API
#   3. Verify answers, filter quality, select shortest correct response
#   4. Output parquet for verl SFT training
#
# Key alignment with KDFlow:
#   - SGLang server (not vLLM)
#   - temperature=0.6, top_p=0.95
#   - n_trajectories=8
#   - Answer verification + quality filtering
#   - Shortest correct response selection
# ============================================================================

set -euo pipefail

export SGLANG_DISABLE_CUDNN_CHECK=1

# --- Configuration ---
TEACHER_MODEL="/path/to/models/Qwen2.5-7B-Instruct"
RAW_DATASET="/path/to/workspace/workspace/dataset/mixed_math_code_10k_with_source"
EASYOPD_ROOT="/path/to/EasyOPD"
EXP_DIR="${EASYOPD_ROOT}/experiments/01_cross_tokenizer_opd"
OUTPUT_PARQUET="${EXP_DIR}/train_data/teacher_sft_train.parquet"
PYTHON="/opt/conda/envs/OpenAgentRL-sglang/bin/python"

# SGLang server config
PORT=30000
DP_SIZE=8
TP_SIZE=1
MEM_FRACTION=0.85
SERVER_LOG="/tmp/sglang_server_teacher_gen.log"

# Generation config (aligned with KDFlow)
TEMPERATURE=0.6
TOP_P=0.95
N_TRAJECTORIES=8
MAX_NEW_TOKENS=4096
MAX_CONCURRENT=256
BATCH_SIZE=2000

# Quality filter config (aligned with KDFlow)
MIN_RESPONSE_LENGTH=20
MAX_RESPONSE_LENGTH=4000
MAX_REPETITION_RATIO=0.30

echo "[$(date)] ============================================================"
echo "[$(date)] Generate Teacher SFT Data (KDFlow-aligned)"
echo "[$(date)] ============================================================"
echo "[$(date)] Teacher model: ${TEACHER_MODEL}"
echo "[$(date)] Raw dataset: ${RAW_DATASET}"
echo "[$(date)] Output: ${OUTPUT_PARQUET}"
echo "[$(date)] Trajectories per question: ${N_TRAJECTORIES}"
echo "[$(date)] Temperature: ${TEMPERATURE}, Top-p: ${TOP_P}"
echo "[$(date)] Max new tokens: ${MAX_NEW_TOKENS}"
echo "[$(date)] ============================================================"

# --- Backup existing data ---
if [ -f "${OUTPUT_PARQUET}" ]; then
    BACKUP="${OUTPUT_PARQUET}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "[$(date)] Backing up existing parquet to: ${BACKUP}"
    mv "${OUTPUT_PARQUET}" "${BACKUP}"
fi

# --- Step 1: Start SGLang server ---
echo "[$(date)] Starting SGLang server (DP=${DP_SIZE}, TP=${TP_SIZE}, port=${PORT})..."

# Clean up old processes
OLD_PIDS=$(lsof -ti :${PORT} 2>/dev/null || true)
if [ -n "${OLD_PIDS}" ]; then
    echo "[$(date)] Killing old processes on port ${PORT}: ${OLD_PIDS}"
    echo "${OLD_PIDS}" | xargs kill -9 2>/dev/null || true
    sleep 3
fi
pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
sleep 2

${PYTHON} -m sglang.launch_server \
    --model-path ${TEACHER_MODEL} \
    --dp-size ${DP_SIZE} \
    --tp-size ${TP_SIZE} \
    --port ${PORT} \
    --mem-fraction-static ${MEM_FRACTION} \
    --trust-remote-code \
    > ${SERVER_LOG} 2>&1 &

SERVER_PID=$!
echo "[$(date)] Server PID: ${SERVER_PID}"

cleanup() {
    echo "[$(date)] Shutting down server (PID=${SERVER_PID})..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server
echo "[$(date)] Waiting for server to be ready..."
SERVER_READY=0
for i in $(seq 1 120); do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[$(date)] Server process died! Check ${SERVER_LOG}"
        tail -30 ${SERVER_LOG}
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/health 2>/dev/null || echo "000")
    if [ "${HTTP_CODE}" = "200" ]; then
        echo "[$(date)] Server is ready! (waited $((i*5))s)"
        SERVER_READY=1
        break
    fi
    if [ $((i % 12)) -eq 0 ]; then
        echo "[$(date)]   Still waiting... ($((i*5))s elapsed)"
    fi
    sleep 5
done

if [ ${SERVER_READY} -ne 1 ]; then
    echo "[$(date)] Server failed to start after 600s!"
    tail -30 ${SERVER_LOG}
    exit 1
fi

# --- Step 2: Generate responses and build SFT data ---
echo "[$(date)] Running teacher response generation + filtering..."

${PYTHON} $(dirname "$0")/gen_teacher_sft_data.py \
    --raw_dataset "${RAW_DATASET}" \
    --output_parquet "${OUTPUT_PARQUET}" \
    --base_url "http://127.0.0.1:${PORT}" \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --n_trajectories ${N_TRAJECTORIES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --max_concurrent ${MAX_CONCURRENT} \
    --batch_size ${BATCH_SIZE} \
    --min_response_length ${MIN_RESPONSE_LENGTH} \
    --max_response_length ${MAX_RESPONSE_LENGTH} \
    --max_repetition_ratio ${MAX_REPETITION_RATIO}

echo "[$(date)] ============================================================"
echo "[$(date)] Teacher SFT data generation complete!"
echo "[$(date)] Output: ${OUTPUT_PARQUET}"
echo "[$(date)] ============================================================"
