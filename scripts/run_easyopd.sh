#!/bin/bash
# Copyright 2026 EasyOPD Contributors
#
# EasyOPD Unified Launch Script
#
# Usage:
#   bash scripts/run_easyopd.sh --method gkd --config easyopd/config/gkd.yaml
#   bash scripts/run_easyopd.sh --method simple
#   bash scripts/run_easyopd.sh --list-methods
#
# Environment variables:
#   EASYOPD_METHOD    - Method name (alternative to --method)
#   EASYOPD_CONFIG    - Config path (alternative to --config)
#   NPROC_PER_NODE    - Number of GPUs per node (default: auto-detect)
#   NNODES            - Number of nodes (default: 1)
#   MASTER_ADDR       - Master node address (default: localhost)
#   MASTER_PORT       - Master node port (default: 29500)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

METHOD="${EASYOPD_METHOD:-}"
CONFIG="${EASYOPD_CONFIG:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
DRY_RUN=false
LIST_METHODS=false
EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --nproc-per-node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --list-methods)
            LIST_METHODS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# List methods mode
# ---------------------------------------------------------------------------
if [ "$LIST_METHODS" = true ]; then
    cd "$PROJECT_ROOT"
    python scripts/run_easyopd.py --list-methods
    exit 0
fi

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ -z "$METHOD" ]; then
    echo "Error: --method is required."
    echo "Usage: bash scripts/run_easyopd.sh --method <name> [--config <path>]"
    echo "       bash scripts/run_easyopd.sh --list-methods"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  EasyOPD Training Launch"
echo "============================================================"
echo "  Method:          $METHOD"
echo "  Config:          ${CONFIG:-<default>}"
echo "  GPUs per node:   $NPROC_PER_NODE"
echo "  Nodes:           $NNODES"
echo "  Master:          $MASTER_ADDR:$MASTER_PORT"
echo "============================================================"

# Build Python command
CMD="python scripts/run_easyopd.py --method $METHOD"

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD="$CMD ${EXTRA_ARGS[*]}"
fi

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
echo "Running: $CMD"
echo ""

exec $CMD
