#!/usr/bin/env bash
# Lightning-OPD Step 1: Generate SFT training data using teacher model
#
# This is a thin wrapper that delegates to EasyOPD's existing
# data_preprocess / vLLM rollout infrastructure.
#
# Usage:
#   bash examples/lightning_opd_trainer/tools/generate_sft_data.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LIGHTNING_OPD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

TEACHER_MODEL="${LIGHTNING_OPD_TEACHER_MODEL:-<SET_LIGHTNING_OPD_TEACHER_MODEL>}"
SFT_PROMPTS="${LIGHTNING_OPD_SFT_PROMPTS:-<SET_LIGHTNING_OPD_SFT_PROMPTS>}"
OUTPUT_PARQUET="${LIGHTNING_OPD_OUT:-./data/sft_generated/train.parquet}"
TEACHER_URL="${LIGHTNING_OPD_TEACHER_CHAT_URL:-${LIGHTNING_OPD_TEACHER_URL:-http://127.0.0.1:8000/v1/chat/completions}}"
MAX_TOKENS="${LIGHTNING_OPD_MAX_TOKENS:-4096}"
TEMPERATURE="${LIGHTNING_OPD_TEMPERATURE:-0.6}"
CONCURRENCY="${LIGHTNING_OPD_CONCURRENCY:-16}"

if [ "${LIGHTNING_OPD_DRYRUN:-}" = "true" ]; then
    echo "=== Lightning-OPD Generate SFT Data Dry-Run ==="
    echo "TEACHER_MODEL: $TEACHER_MODEL"
    echo "SFT_PROMPTS:   $SFT_PROMPTS"
    echo "TEACHER_URL:   $TEACHER_URL"
    echo "OUTPUT_PARQUET: $OUTPUT_PARQUET"
    echo ""
    echo "python3 -m easyopd.methods.lightning_opd.data_curation.generate_responses \\"
    echo "    --input-prompts $SFT_PROMPTS \\"
    echo "    --output-parquet $OUTPUT_PARQUET \\"
    echo "    --model $TEACHER_MODEL \\"
    echo "    --endpoint $TEACHER_URL \\"
    echo "    --max-tokens $MAX_TOKENS \\"
    echo "    --temperature $TEMPERATURE \\"
    echo "    --concurrency $CONCURRENCY"
    exit 0
fi

if [[ "$TEACHER_MODEL" == *"<"* ]] || [[ "$SFT_PROMPTS" == *"<"* ]]; then
    echo "Error: Set LIGHTNING_OPD_TEACHER_MODEL and LIGHTNING_OPD_SFT_PROMPTS" >&2
    exit 1
fi

python3 -m easyopd.methods.lightning_opd.data_curation.generate_responses \
    --input-prompts "$SFT_PROMPTS" \
    --output-parquet "$OUTPUT_PARQUET" \
    --model "$TEACHER_MODEL" \
    --endpoint "$TEACHER_URL" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY"
