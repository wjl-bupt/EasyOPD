#!/usr/bin/env bash
# Lightning-OPD Step 6: Convert Megatron checkpoint to HuggingFace format
#
# Usage:
#   bash examples/lightning_opd_trainer/tools/convert_megatron_to_hf.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LIGHTNING_OPD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

MEGATRON_CKPT_DIR="${MEGATRON_CKPT_DIR:-<SET_MEGATRON_CKPT_DIR>}"
HF_OUTPUT_DIR="${HF_OUTPUT_DIR:-<SET_HF_OUTPUT_DIR>}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-<SET_ORIGIN_HF_DIR>}"

if [ "${LIGHTNING_OPD_DRYRUN:-}" = "true" ]; then
    echo "=== Lightning-OPD Megatron→HF Dry-Run ==="
    echo "MEGATRON_CKPT_DIR: $MEGATRON_CKPT_DIR"
    echo "HF_OUTPUT_DIR:     $HF_OUTPUT_DIR"
    echo "ORIGIN_HF_DIR:     $ORIGIN_HF_DIR"
    echo ""
    echo "# verl already provides checkpoint conversion utilities."
    echo "# See verl/trainer/main_ppo.py checkpoint conversion options."
    exit 0
fi

if [[ "$MEGATRON_CKPT_DIR" == *"<"* ]] || [[ "$HF_OUTPUT_DIR" == *"<"* ]] || [[ "$ORIGIN_HF_DIR" == *"<"* ]]; then
    echo "Error: Set MEGATRON_CKPT_DIR, HF_OUTPUT_DIR, and ORIGIN_HF_DIR" >&2
    exit 1
fi

echo "Error: No generic Megatron→HF converter is wired for this checkout yet." >&2
echo "Use the backend-specific verl checkpoint export path documented by your training backend." >&2
exit 1
