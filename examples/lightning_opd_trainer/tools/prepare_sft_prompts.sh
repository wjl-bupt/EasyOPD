#!/usr/bin/env bash
# Lightning-OPD Step 0: Prepare SFT prompts from HF dataset
#
# Usage:
#   bash examples/lightning_opd_trainer/tools/prepare_sft_prompts.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LIGHTNING_OPD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

HF_DATASET="${LIGHTNING_OPD_HF_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
OUTPUT="${LIGHTNING_OPD_OUT:-./data/prompts/openthoughts3_300k.jsonl}"
NUM_SAMPLES="${LIGHTNING_OPD_NUM_SAMPLES:-300000}"
INPUT_PARQUET="${LIGHTNING_OPD_INPUT_PARQUET:-}"

INPUT_ARG=""
if [ -n "$INPUT_PARQUET" ]; then
    INPUT_ARG="--input-parquet $INPUT_PARQUET"
fi

if [ "${LIGHTNING_OPD_DRYRUN:-}" = "true" ]; then
    echo "=== Lightning-OPD Prepare SFT Prompts Dry-Run ==="
    echo "python3 -m easyopd.methods.lightning_opd.data_curation.prompt_prep \\"
    echo "    --hf-dataset $HF_DATASET \\"
    echo "    --output $OUTPUT \\"
    echo "    --num-samples $NUM_SAMPLES \\"
    echo "    $INPUT_ARG"
    exit 0
fi

python3 -m easyopd.methods.lightning_opd.data_curation.prompt_prep \
    --hf-dataset "$HF_DATASET" \
    --output "$OUTPUT" \
    --num-samples "$NUM_SAMPLES" \
    $INPUT_ARG
