#!/bin/bash
# Auto pipeline: Wait for SimCT training -> Merge -> Evaluate all Phi-4-mini models
set -euo pipefail

WORKDIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD"
PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
RAY="/opt/conda/envs/OpenAgentRL-sj/bin/ray"
SIMCT_LOG="/tmp/easyopd_simct_phi4mini_final.log"
SIMCT_CKPT="${WORKDIR}/experiments/benchmark/checkpoints/simct_phi4mini"
SIMPLE_MERGED="${WORKDIR}/experiments/benchmark/checkpoints/simple_phi4mini/merged_hf"
BASE_MODEL="/root/workspace/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6/checkpoint-40"
RESULTS_DIR="${WORKDIR}/experiments/benchmark/results"

echo "[$(date)] Starting auto pipeline..."

# ===== Step 1: Wait for SimCT training to finish =====
echo "[$(date)] Waiting for SimCT training to complete..."
while true; do
    if ! ps aux | grep "main_ppo" | grep -v grep > /dev/null 2>&1; then
        echo "[$(date)] SimCT training process ended."
        break
    fi
    progress=$(grep "Training Progress" "$SIMCT_LOG" 2>/dev/null | tail -1 || true)
    echo "[$(date)] $progress"
    sleep 120
done

# Check for errors
if grep -q "Traceback" "$SIMCT_LOG" 2>/dev/null; then
    echo "[$(date)] ERROR: SimCT training failed!"
    grep -A 5 "Traceback" "$SIMCT_LOG" | tail -10
    exit 1
fi
echo "[$(date)] SimCT training completed successfully."

# ===== Step 2: Merge SimCT FSDP checkpoint =====
echo "[$(date)] Merging SimCT FSDP checkpoint..."
SIMCT_STEP=$(ls -d ${SIMCT_CKPT}/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -z "$SIMCT_STEP" ]; then
    echo "[$(date)] ERROR: No SimCT checkpoint found!"
    exit 1
fi

SIMCT_MERGED="${SIMCT_CKPT}/merged_hf"
cd "$WORKDIR"
$PYTHON experiments/benchmark/merge_fsdp.py \
    --ckpt_dir "${SIMCT_STEP}/actor" \
    --base_model "$BASE_MODEL" \
    --output_dir "$SIMCT_MERGED"

echo "[$(date)] SimCT merge complete: $SIMCT_MERGED"

# ===== Step 3: Stop Ray (free GPU memory for evaluation) =====
echo "[$(date)] Stopping Ray for evaluation..."
$RAY stop --force 2>/dev/null || true
sleep 5

# ===== Step 4: Evaluate Base Phi-4-mini =====
echo "[$(date)] Evaluating Base Phi-4-mini..."
cd "$WORKDIR"
$PYTHON experiments/benchmark/evaluate_model.py \
    --model_path "$BASE_MODEL" \
    --model_name "base_phi4mini" \
    --output_dir "$RESULTS_DIR" \
    --benchmarks "math500,gsm8k" \
    --tensor_parallel_size 1

# ===== Step 5: Evaluate Simple (Phi-4-mini) =====
echo "[$(date)] Evaluating Simple (Phi-4-mini)..."
$PYTHON experiments/benchmark/evaluate_model.py \
    --model_path "$SIMPLE_MERGED" \
    --model_name "simple_phi4mini" \
    --output_dir "$RESULTS_DIR" \
    --benchmarks "math500,gsm8k" \
    --tensor_parallel_size 1

# ===== Step 6: Evaluate SimCT (Phi-4-mini) =====
echo "[$(date)] Evaluating SimCT (Phi-4-mini)..."
$PYTHON experiments/benchmark/evaluate_model.py \
    --model_path "$SIMCT_MERGED" \
    --model_name "simct_phi4mini" \
    --output_dir "$RESULTS_DIR" \
    --benchmarks "math500,gsm8k" \
    --tensor_parallel_size 1

# ===== Step 7: Update CSV =====
echo "[$(date)] Updating benchmark_results.csv..."
$PYTHON -c "
import json, csv, os

results_dir = '${RESULTS_DIR}'
csv_path = os.path.join(results_dir, 'benchmark_results.csv')

# Load all results
models = {
    'Base (Qwen2.5-1.5B-Instruct)': 'base_qwen2.5-1.5b',  # keep existing
    'GRPO (Qwen2.5-1.5B)': 'grpo',  # keep existing
    'Base (Phi-4-mini SFT)': 'base_phi4mini',
    'Simple (Phi-4-mini)': 'simple_phi4mini',
    'SimCT (Phi-4-mini)': 'simct_phi4mini',
}

rows = []
for display_name, file_prefix in models.items():
    summary_path = os.path.join(results_dir, f'{file_prefix}_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        math500 = data.get('results', {}).get('math500', {})
        gsm8k = data.get('results', {}).get('gsm8k', {})
        rows.append({
            'method': display_name,
            'math500_accuracy': math500.get('accuracy', ''),
            'math500_correct': math500.get('correct', ''),
            'math500_total': math500.get('total', ''),
            'gsm8k_accuracy': gsm8k.get('accuracy', ''),
            'gsm8k_correct': gsm8k.get('correct', ''),
            'gsm8k_total': gsm8k.get('total', ''),
        })
    else:
        print(f'Warning: {summary_path} not found, skipping {display_name}')

# Write CSV
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['method', 'math500_accuracy', 'math500_correct', 'math500_total', 'gsm8k_accuracy', 'gsm8k_correct', 'gsm8k_total'])
    writer.writeheader()
    writer.writerows(rows)

print(f'Updated {csv_path} with {len(rows)} entries')
for r in rows:
    print(f\"  {r['method']}: MATH-500={r['math500_accuracy']}%, GSM8K={r['gsm8k_accuracy']}%\")
"

echo "[$(date)] ===== ALL DONE ====="
echo "[$(date)] Results saved to: ${RESULTS_DIR}/benchmark_results.csv"
