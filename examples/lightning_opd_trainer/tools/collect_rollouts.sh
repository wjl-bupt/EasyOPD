#!/usr/bin/env bash
# Lightning-OPD Step 3: Collect student rollouts on OPD prompts
#
# Uses the SFT-trained student model to generate responses on OPD
# prompts (e.g. DAPO-Math-17k).
#
# Usage:
#   bash examples/lightning_opd_trainer/tools/collect_rollouts.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LIGHTNING_OPD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

SFT_CHECKPOINT="${LIGHTNING_OPD_SFT_CHECKPOINT:-<SET_LIGHTNING_OPD_SFT_CHECKPOINT>}"
OPD_PROMPTS="${LIGHTNING_OPD_OPD_PROMPTS:-<SET_LIGHTNING_OPD_OPD_PROMPTS>}"
OUTPUT_PARQUET="${LIGHTNING_OPD_OUT:-./data/rollouts/rollouts.parquet}"
STUDENT_URL="${LIGHTNING_OPD_STUDENT_URL:-http://127.0.0.1:8000/v1/chat/completions}"
MAX_TOKENS="${LIGHTNING_OPD_MAX_TOKENS:-4096}"
TEMPERATURE="${LIGHTNING_OPD_TEMPERATURE:-0.6}"
CONCURRENCY="${LIGHTNING_OPD_CONCURRENCY:-16}"

if [ "${LIGHTNING_OPD_DRYRUN:-}" = "true" ]; then
    echo "=== Lightning-OPD Collect Rollouts Dry-Run ==="
    echo "SFT_CHECKPOINT: $SFT_CHECKPOINT"
    echo "OPD_PROMPTS:    $OPD_PROMPTS"
    echo "STUDENT_URL:    $STUDENT_URL"
    echo "OUTPUT_PARQUET: $OUTPUT_PARQUET"
    echo ""
    echo "python3 -m easyopd.methods.lightning_opd.data_curation.generate_responses \\"
    echo "    --input-prompts $OPD_PROMPTS \\"
    echo "    --output-parquet $OUTPUT_PARQUET \\"
    echo "    --model $SFT_CHECKPOINT \\"
    echo "    --endpoint $STUDENT_URL \\"
    echo "    --max-tokens $MAX_TOKENS \\"
    echo "    --temperature $TEMPERATURE \\"
    echo "    --concurrency $CONCURRENCY"
    exit 0
fi

if [[ "$SFT_CHECKPOINT" == *"<"* ]] || [[ "$OPD_PROMPTS" == *"<"* ]]; then
    echo "Error: Set LIGHTNING_OPD_SFT_CHECKPOINT and LIGHTNING_OPD_OPD_PROMPTS" >&2
    exit 1
fi

python3 -m easyopd.methods.lightning_opd.data_curation.generate_responses \
    --input-prompts "$OPD_PROMPTS" \
    --output-parquet "$OUTPUT_PARQUET" \
    --model "$SFT_CHECKPOINT" \
    --endpoint "$STUDENT_URL" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY"
