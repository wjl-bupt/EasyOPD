#!/usr/bin/env bash
# Lightning-OPD Step 2: SFT training (paper §3.2 recipe)
#
# Thin wrapper calling verl fsdp_sft_trainer with the Lightning-OPD
# sft.yaml config (3000 steps, lr=8e-5, packing-enabled).
#
# Usage:
#   bash examples/lightning_opd_trainer/tools/run_sft.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LIGHTNING_OPD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

SFT_BASE_MODEL="${LIGHTNING_OPD_SFT_BASE_MODEL:-<SET_LIGHTNING_OPD_SFT_BASE_MODEL>}"
SFT_DATA="${LIGHTNING_OPD_SFT_DATA:-<SET_LIGHTNING_OPD_SFT_DATA>}"
SFT_OUT="${LIGHTNING_OPD_SFT_OUT:-./checkpoint/sft_lightning_opd}"

if [ "${LIGHTNING_OPD_DRYRUN:-}" = "true" ]; then
    echo "=== Lightning-OPD SFT Training Dry-Run ==="
    echo "PROJECT_ROOT:   $PROJECT_ROOT"
    echo "SFT_BASE_MODEL: $SFT_BASE_MODEL"
    echo "SFT_DATA:       $SFT_DATA"
    echo "SFT_OUT:        $SFT_OUT"
    echo ""
    echo "python3 -m verl.trainer.fsdp_sft_trainer \\"
    echo "    --config-path $PROJECT_ROOT/easyopd/config/lightning_opd \\"
    echo "    --config-name sft \\"
    echo "    data.path=$SFT_DATA \\"
    echo "    model.path=$SFT_BASE_MODEL \\"
    echo "    trainer.default_local_dir=$SFT_OUT"
    exit 0
fi

python3 -m verl.trainer.fsdp_sft_trainer \
    --config-path "$PROJECT_ROOT/easyopd/config/lightning_opd" \
    --config-name sft \
    "data.path=$SFT_DATA" \
    "model.path=$SFT_BASE_MODEL" \
    "trainer.default_local_dir=$SFT_OUT" \
    "$@"
