#!/bin/bash
set -e

WORK_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD"
BENCHMARK_DIR="${WORK_DIR}/experiments/benchmark"
CKPT_DIR="${BENCHMARK_DIR}/checkpoints"
RESULTS_DIR="${BENCHMARK_DIR}/results"
BASE_MODEL="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-1.5B-Instruct"
PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"

cd ${WORK_DIR}

echo "============================================"
echo "Step 1: Merge GRPO checkpoint"
echo "============================================"
if [ ! -f "${CKPT_DIR}/grpo/merged_hf/model.safetensors" ]; then
    echo "Merging GRPO checkpoint..."
    ${PYTHON} ${BENCHMARK_DIR}/merge_checkpoint.py \
        --ckpt_dir ${CKPT_DIR}/grpo/global_step_200/actor \
        --output_dir ${CKPT_DIR}/grpo/merged_hf \
        --base_model ${BASE_MODEL}
    echo "GRPO merge completed!"
else
    echo "GRPO already merged, skipping."
fi

echo ""
echo "============================================"
echo "Step 2: Merge SimCT checkpoint"
echo "============================================"
if [ ! -f "${CKPT_DIR}/simct/merged_hf/model.safetensors" ]; then
    echo "Merging SimCT checkpoint..."
    ${PYTHON} ${BENCHMARK_DIR}/merge_checkpoint.py \
        --ckpt_dir ${CKPT_DIR}/simct/global_step_200/actor \
        --output_dir ${CKPT_DIR}/simct/merged_hf \
        --base_model ${BASE_MODEL}
    echo "SimCT merge completed!"
else
    echo "SimCT already merged, skipping."
fi

echo ""
echo "============================================"
echo "Step 3: Evaluate GRPO model"
echo "============================================"
mkdir -p ${RESULTS_DIR}/grpo
if [ ! -f "${RESULTS_DIR}/grpo/grpo_qwen2.5-1.5b_summary.json" ]; then
    echo "Evaluating GRPO model..."
    ${PYTHON} ${BENCHMARK_DIR}/evaluate_model.py \
        --model_path ${CKPT_DIR}/grpo/merged_hf \
        --model_name grpo_qwen2.5-1.5b \
        --output_dir ${RESULTS_DIR}/grpo \
        --benchmarks math500,gsm8k \
        --tensor_parallel_size 1
    echo "GRPO evaluation completed!"
else
    echo "GRPO already evaluated, skipping."
fi

echo ""
echo "============================================"
echo "Step 4: Evaluate Simple model"
echo "============================================"
mkdir -p ${RESULTS_DIR}/simple
if [ ! -f "${RESULTS_DIR}/simple/simple_qwen2.5-1.5b_summary.json" ]; then
    echo "Evaluating Simple model..."
    ${PYTHON} ${BENCHMARK_DIR}/evaluate_model.py \
        --model_path ${CKPT_DIR}/simple/merged_hf \
        --model_name simple_qwen2.5-1.5b \
        --output_dir ${RESULTS_DIR}/simple \
        --benchmarks math500,gsm8k \
        --tensor_parallel_size 1
    echo "Simple evaluation completed!"
else
    echo "Simple already evaluated, skipping."
fi

echo ""
echo "============================================"
echo "Step 5: Evaluate SimCT model"
echo "============================================"
mkdir -p ${RESULTS_DIR}/simct
if [ ! -f "${RESULTS_DIR}/simct/simct_qwen2.5-1.5b_summary.json" ]; then
    echo "Evaluating SimCT model..."
    ${PYTHON} ${BENCHMARK_DIR}/evaluate_model.py \
        --model_path ${CKPT_DIR}/simct/merged_hf \
        --model_name simct_qwen2.5-1.5b \
        --output_dir ${RESULTS_DIR}/simct \
        --benchmarks math500,gsm8k \
        --tensor_parallel_size 1
    echo "SimCT evaluation completed!"
else
    echo "SimCT already evaluated, skipping."
fi

echo ""
echo "============================================"
echo "All done! Final Summary"
echo "============================================"
echo ""
echo "Base model results:"
cat ${RESULTS_DIR}/base_qwen2.5-1.5b_summary.json 2>/dev/null || echo "  Not available"
echo ""
echo "GRPO results:"
cat ${RESULTS_DIR}/grpo/grpo_qwen2.5-1.5b_summary.json 2>/dev/null || echo "  Not available"
echo ""
echo "Simple results:"
cat ${RESULTS_DIR}/simple/simple_qwen2.5-1.5b_summary.json 2>/dev/null || echo "  Not available"
echo ""
echo "SimCT results:"
cat ${RESULTS_DIR}/simct/simct_qwen2.5-1.5b_summary.json 2>/dev/null || echo "  Not available"
