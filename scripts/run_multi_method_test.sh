#!/bin/bash
# =============================================================================
# EasyOPD Multi-Method Real Training Test
# Runs multiple methods sequentially on 8 GPUs with real model and data.
# Each method runs for 5 steps with GRPO + EasyOPD hook.
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/path/to/EasyOPD"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export RAY_ADDRESS=auto

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
STUDENT="/path/to/workspace/workspace/models/Qwen2.5-1.5B-Instruct"
OUTPUT_BASE="/tmp/easyopd_real_test"
RESULTS_FILE="${OUTPUT_BASE}/multi_method_results.txt"

mkdir -p "$OUTPUT_BASE"
echo "EasyOPD Multi-Method Real Training Results" > "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "Student: $STUDENT" >> "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"

# Methods to test (all use GRPO + EasyOPD hook, no separate teacher vLLM needed)
METHODS=("gkd" "sod" "opcd" "g_opd" "sdpo" "echo_kd" "opsa" "ropd" "vision_opd")

PASS=0
FAIL=0

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Testing method: $METHOD"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    LOG_FILE="${OUTPUT_BASE}/${METHOD}.log"
    START_TIME=$(date +%s)
    
    $PYTHON -m verl.trainer.main_ppo \
        data.train_files="['test_data/train.parquet']" \
        data.val_files="['test_data/test.parquet']" \
        data.train_batch_size=8 \
        data.max_prompt_length=256 \
        data.max_response_length=128 \
        data.truncation=right \
        data.prompt_key=prompt \
        actor_rollout_ref.model.path=$STUDENT \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=5e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.max_num_seqs=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        +easyopd.method.name=$METHOD \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.total_training_steps=5 \
        trainer.save_freq=999 \
        trainer.test_freq=999 \
        'trainer.logger=["console"]' \
        trainer.project_name=easyopd_real_test \
        trainer.experiment_name=${METHOD}_real \
        trainer.default_local_dir=${OUTPUT_BASE}/${METHOD} \
        custom_reward_function.path=examples/simple/reward_real.py \
        custom_reward_function.name=compute_score \
        > "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Check if training completed successfully (look for step:5 in log)
    STEPS_DONE=$(grep -c "step:" "$LOG_FILE" 2>/dev/null || echo "0")
    
    if [ $EXIT_CODE -eq 0 ] && [ "$STEPS_DONE" -ge 5 ]; then
        # Extract loss from last step
        LAST_LOSS=$(grep "step:5" "$LOG_FILE" 2>/dev/null | grep -o "actor/pg_loss:[^ ]*" | cut -d: -f2)
        LAST_GRAD=$(grep "step:5" "$LOG_FILE" 2>/dev/null | grep -o "actor/grad_norm:[^ ]*" | cut -d: -f2)
        echo "  ✅ PASS: $METHOD (${DURATION}s, ${STEPS_DONE} steps, loss=${LAST_LOSS}, grad=${LAST_GRAD})"
        echo "PASS $METHOD ${DURATION}s steps=${STEPS_DONE} loss=${LAST_LOSS} grad=${LAST_GRAD}" >> "$RESULTS_FILE"
        ((PASS++))
    else
        echo "  ❌ FAIL: $METHOD (exit=$EXIT_CODE, ${DURATION}s, ${STEPS_DONE} steps)"
        echo "  Last 10 lines of log:"
        tail -10 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
        echo "FAIL $METHOD exit=$EXIT_CODE ${DURATION}s steps=${STEPS_DONE}" >> "$RESULTS_FILE"
        ((FAIL++))
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    FINAL RESULTS                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Passed:  $PASS / ${#METHODS[@]}                                          ║"
echo "║  Failed:  $FAIL / ${#METHODS[@]}                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: ${OUTPUT_BASE}/<method>.log"
cat "$RESULTS_FILE"
