#!/usr/bin/env bash
# lightning_opd: Prepare training data (Phase 1 tokenize + Phase 2 teacher logprobs)
#
# Usage:
#   bash examples/lightning_opd_trainer/tools/prepare_data.sh
#
# Dry-run:
#   LIGHTNING_OPD_DRYRUN=true bash examples/lightning_opd_trainer/tools/prepare_data.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LIGHTNING_OPD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

TOKENIZER="${LIGHTNING_OPD_TOKENIZER:-<SET_LIGHTNING_OPD_TOKENIZER>}"
ROLLOUTS="${LIGHTNING_OPD_ROLLOUTS:-<SET_LIGHTNING_OPD_ROLLOUTS>}"
TEACHER_URL="${LIGHTNING_OPD_TEACHER_URL:-http://127.0.0.1:8000/v1/completions}"
OUTPUT_DIR="${LIGHTNING_OPD_OUT:-./data/lightning_opd}"
MAX_RESPONSE_LEN="${LIGHTNING_OPD_MAX_RESPONSE_LEN:-4096}"
CONCURRENCY="${LIGHTNING_OPD_CONCURRENCY:-64}"
SFT_TEACHER="${LIGHTNING_OPD_SFT_TEACHER:-}"
OPD_TEACHER="${LIGHTNING_OPD_OPD_TEACHER:-}"

CONSISTENCY_ARGS=()
CONSISTENCY_STATUS="skipped"
if [ -n "$SFT_TEACHER" ] && [ -n "$OPD_TEACHER" ]; then
    if python3 - "$SFT_TEACHER" "$OPD_TEACHER" <<'PY'
import sys
from easyopd.methods.lightning_opd.teacher_consistency import check_teacher_consistency

check_teacher_consistency(sys.argv[1], sys.argv[2])
print("CONSISTENT")
PY
    then
        CONSISTENCY_STATUS="OK (same model)"
        CONSISTENCY_ARGS=(--sft-teacher-id "$SFT_TEACHER" --opd-teacher-id "$OPD_TEACHER")
    else
        echo "Teacher consistency check failed for prepare_data.sh." >&2
        exit 1
    fi
fi

if [ "${LIGHTNING_OPD_DRYRUN:-}" = "true" ]; then
    echo "=== lightning_opd Prepare Data Dry-Run ==="
    echo "PROJECT_ROOT:      $PROJECT_ROOT"
    echo "TOKENIZER:         $TOKENIZER"
    echo "ROLLOUTS:          $ROLLOUTS"
    echo "TEACHER_URL:       $TEACHER_URL"
    echo "OUTPUT_DIR:        $OUTPUT_DIR"
    echo "MAX_RESPONSE_LEN:  $MAX_RESPONSE_LEN"
    echo "CONCURRENCY:       $CONCURRENCY"
    echo "SFT_TEACHER:       ${SFT_TEACHER:-<not set>}"
    echo "OPD_TEACHER:       ${OPD_TEACHER:-<not set>}"
    echo "Teacher consistency: $CONSISTENCY_STATUS"
    echo ""
    echo "# Phase 1 (CPU): Tokenize student rollouts"
    echo "python3 -m easyopd.methods.lightning_opd.data_curation.prepare \\"
    echo "    --tokenizer-path $TOKENIZER \\"
    echo "    --input-parquet $ROLLOUTS \\"
    echo "    --output-dir $OUTPUT_DIR \\"
    echo "    --max-response-len $MAX_RESPONSE_LEN \\"
    if [ "${#CONSISTENCY_ARGS[@]}" -gt 0 ]; then
        echo "    ${CONSISTENCY_ARGS[*]}"
    fi
    echo ""
    echo "# Phase 2 (GPU): Compute teacher logprobs"
    echo "python3 -m easyopd.methods.lightning_opd.data_curation.prepare \\"
    echo "    --tokenizer-path $TOKENIZER \\"
    echo "    --input-parquet $ROLLOUTS \\"
    echo "    --output-dir $OUTPUT_DIR \\"
    echo "    --compute-teacher-logprobs \\"
    echo "    --teacher-url $TEACHER_URL \\"
    echo "    --concurrency $CONCURRENCY \\"
    if [ "${#CONSISTENCY_ARGS[@]}" -gt 0 ]; then
        echo "    ${CONSISTENCY_ARGS[*]}"
    fi
    exit 0
fi

# Phase 1: tokenize
python3 -m easyopd.methods.lightning_opd.data_curation.prepare \
    --tokenizer-path "$TOKENIZER" \
    --input-parquet "$ROLLOUTS" \
    --output-dir "$OUTPUT_DIR" \
    --max-response-len "$MAX_RESPONSE_LEN" \
    "${CONSISTENCY_ARGS[@]}"

# Phase 2: teacher logprobs
python3 -m easyopd.methods.lightning_opd.data_curation.prepare \
    --tokenizer-path "$TOKENIZER" \
    --input-parquet "$ROLLOUTS" \
    --output-dir "$OUTPUT_DIR" \
    --compute-teacher-logprobs \
    --teacher-url "$TEACHER_URL" \
    --concurrency "$CONCURRENCY" \
    "${CONSISTENCY_ARGS[@]}"
