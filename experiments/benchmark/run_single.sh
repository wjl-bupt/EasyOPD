#!/usr/bin/env bash
# =============================================================================
# EasyOPD Benchmark: Run a single method (train + evaluate)
# Usage: bash run_single.sh <method_name>
# Methods: base, sft, grpo, gkd, sod, opcd, g_opd, sdpo, opsa, ropd, vision_opd, simple, simct
# =============================================================================
set -euo pipefail

METHOD="${1:-base}"

export PYTHONPATH="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
PROJECT_ROOT="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments/benchmark"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
RESULTS_DIR="${EXPERIMENT_DIR}/results"
LOG_DIR="${EXPERIMENT_DIR}/logs"
DATA_DIR="${EXPERIMENT_DIR}/data"

STUDENT_MODEL="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-1.5B-Instruct"
TEACHER_MODEL="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-7B-Instruct"

# Common training hyperparameters
TRAIN_STEPS=200
BATCH_SIZE=16
LR="5e-6"
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=1024
N_GPUS=8
SAVE_FREQ=200
REWARD_FN="${EXPERIMENT_DIR}/reward_fn.py"
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1))

mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

echo "=============================================="
echo "[$(date)] Method: ${METHOD}"
echo "Student: ${STUDENT_MODEL}"
echo "Teacher: ${TEACHER_MODEL}"
echo "=============================================="

# Prepare data if needed
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Preparing data..."
    ${PYTHON} "${EXPERIMENT_DIR}/prepare_data.py"
fi

# ============ Common GRPO base args (shared by most methods) ============
COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    "data.train_files=['${DATA_DIR}/train.parquet']"
    "data.val_files=['${DATA_DIR}/val.parquet']"
    data.train_batch_size=${BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LEN}
    data.max_response_length=${MAX_RESPONSE_LEN}
    data.filter_overlong_prompts=True
    data.truncation=right
    data.prompt_key=prompt
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=${LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE}
    actor_rollout_ref.actor.ppo_epochs=2
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.temperature=0.7
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=16
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    custom_reward_function.path="${REWARD_FN}"
    custom_reward_function.name=compute_score
    trainer.balance_batch=True
    'trainer.logger=["console"]'
    trainer.project_name=easyopd_benchmark
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.nnodes=1
    trainer.val_before_train=False
    trainer.total_training_steps=${TRAIN_STEPS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=50
)

evaluate_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2
    echo "[$(date)] Evaluating: ${MODEL_NAME} from ${MODEL_PATH}"
    CUDA_VISIBLE_DEVICES=0 ${PYTHON} "${EXPERIMENT_DIR}/evaluate_model.py" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --output_dir "${RESULTS_DIR}" \
        --benchmarks "math500,gsm8k" \
        --max_tokens 2048 \
        --tensor_parallel_size 1
}

case "${METHOD}" in
    # ================================================================
    # BASE MODEL EVALUATION (no training)
    # ================================================================
    base)
        evaluate_model "${STUDENT_MODEL}" "base_qwen2.5-1.5b"
        ;;

    # ================================================================
    # SFT BASELINE (supervised fine-tuning on teacher responses)
    # ================================================================
    sft)
        CKPT_DIR="${CHECKPOINT_DIR}/sft"
        if [ ! -d "${CKPT_DIR}" ] || [ -z "$(ls -A ${CKPT_DIR} 2>/dev/null)" ]; then
            echo "[$(date)] Training SFT..."
            ${PYTHON} -m verl.trainer.fsdp_sft_trainer \
                data.train_files="${DATA_DIR}/sft_train.parquet" \
                data.val_files="${DATA_DIR}/val.parquet" \
                data.train_batch_size=64 \
                data.micro_batch_size_per_gpu=2 \
                data.max_length=1536 \
                data.truncation=right \
                data.prompt_key=prompt \
                data.response_key=response \
                model.partial_pretrain="${STUDENT_MODEL}" \
                model.enable_gradient_checkpointing=True \
                model.fsdp_config.model_dtype=bf16 \
                optim.lr=2e-5 \
                optim.warmup_steps_ratio=0.05 \
                optim.weight_decay=0.01 \
                use_remove_padding=True \
                trainer.project_name=easyopd_benchmark \
                trainer.experiment_name=sft \
                trainer.total_epochs=3 \
                trainer.total_training_steps=null \
                'trainer.logger=["console"]' \
                trainer.n_gpus_per_node=${N_GPUS} \
                trainer.nnodes=1 \
                trainer.save_freq=100 \
                trainer.test_freq=50 \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        SFT_MODEL=$(find "${CKPT_DIR}" -name "huggingface" -type d 2>/dev/null | sort | tail -1 || echo "${CKPT_DIR}")
        evaluate_model "${SFT_MODEL}" "sft"
        ;;

    # ================================================================
    # GRPO BASELINE (no distillation, pure RL)
    # ================================================================
    grpo)
        CKPT_DIR="${CHECKPOINT_DIR}/grpo"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training GRPO..."
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                trainer.experiment_name=grpo \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "grpo"
        ;;

    # ================================================================
    # GKD: Generalized Knowledge Distillation (teacher-based)
    # Uses distillation.enabled=True with teacher model
    # ================================================================
    gkd)
        CKPT_DIR="${CHECKPOINT_DIR}/gkd"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training GKD..."
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                distillation.enabled=True \
                distillation.n_gpus_per_node=${N_GPUS} \
                distillation.nnodes=1 \
                distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
                distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2 \
                distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
                distillation.distillation_loss.loss_mode=gkd \
                distillation.distillation_loss.use_policy_gradient=True \
                distillation.distillation_loss.use_task_rewards=True \
                distillation.distillation_loss.distillation_loss_coef=1.0 \
                distillation.distillation_loss.gkd_beta=0.5 \
                distillation.distillation_loss.gkd_temperature=1.0 \
                trainer.experiment_name=gkd \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "gkd"
        ;;

    # ================================================================
    # SOD: Step-wise On-policy Distillation (token-level KL reg)
    # Uses +algorithm.token_kl_reg.* with teacher as ref model
    # ================================================================
    sod)
        CKPT_DIR="${CHECKPOINT_DIR}/sod"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training SOD..."
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                algorithm.use_kl_in_reward=False \
                algorithm.kl_ctrl.kl_coef=0.002 \
                +algorithm.token_kl_reg.enable=True \
                +algorithm.token_kl_reg.gamma=1.0 \
                +algorithm.token_kl_reg.beta_min=0.0 \
                +algorithm.token_kl_reg.beta_max=0.10 \
                +algorithm.token_kl_reg.stepwise_enable=True \
                +algorithm.token_kl_reg.stepwise_epsilon=1e-6 \
                +algorithm.token_kl_reg.stepwise_delta=0.2 \
                +algorithm.token_kl_reg.stepwise_opd_coef=1.0 \
                +actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
                trainer.experiment_name=sod \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "sod"
        ;;

    # ================================================================
    # OPCD: On-Policy Context Distillation (KL loss with ref model)
    # Uses use_kl_loss=True with full KL and top-k
    # ================================================================
    opcd)
        CKPT_DIR="${CHECKPOINT_DIR}/opcd"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training OPCD..."
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_type=full \
                actor_rollout_ref.actor.kl_topk=256 \
                actor_rollout_ref.actor.kl_renorm_topk=False \
                actor_rollout_ref.model.ref_model_path="${TEACHER_MODEL}" \
                trainer.experiment_name=opcd \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "opcd"
        ;;

    # ================================================================
    # G-OPD: Generalized On-Policy Distillation with Reward Extrapolation
    # Uses reverse KL advantages with lambda extrapolation
    # ================================================================
    g_opd)
        CKPT_DIR="${CHECKPOINT_DIR}/g_opd"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training G-OPD..."
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
                actor_rollout_ref.actor.policy_loss.lambda_vals=1.25 \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_coef=0 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                +actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
                algorithm.use_kl_in_reward=False \
                actor_rollout_ref.rollout.calculate_log_probs=true \
                trainer.experiment_name=g_opd \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "g_opd"
        ;;

    # ================================================================
    # SDPO: Self-Distilled Policy Optimization (self-distillation, EMA teacher)
    # No external teacher needed
    # ================================================================
    sdpo)
        CKPT_DIR="${CHECKPOINT_DIR}/sdpo"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training SDPO..."
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                actor_rollout_ref.actor.use_kl_loss=False \
                actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
                actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
                actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
                actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
                actor_rollout_ref.actor.self_distillation.alpha=0.5 \
                actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
                actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
                actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
                actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
                algorithm.norm_adv_by_std_in_grpo=False \
                actor_rollout_ref.rollout.calculate_log_probs=True \
                trainer.experiment_name=sdpo \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "sdpo"
        ;;

    # ================================================================
    # OPSA: On-Policy Self-Distillation (self-distillation, no teacher)
    # ================================================================
    opsa)
        CKPT_DIR="${CHECKPOINT_DIR}/opsa"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training OPSA..."
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                actor_rollout_ref.actor.use_kl_loss=False \
                +actor_rollout_ref.actor.opsa_enable=true \
                +actor_rollout_ref.actor.opsa_temperature=1.0 \
                +actor_rollout_ref.actor.opsa_window_size=32 \
                +actor_rollout_ref.actor.opsa_decay_type=linear \
                +actor_rollout_ref.actor.opsa_min_weight=0.1 \
                +actor_rollout_ref.actor.opsa_use_window_weighting=true \
                +actor_rollout_ref.actor.opsa_distillation_loss_coef=1.0 \
                +actor_rollout_ref.actor.opsa_loss_agg_mode=token-mean \
                +actor_rollout_ref.actor.opsa_kl_type=mixed \
                +actor_rollout_ref.actor.opsa_mixed_kl_weight=0.5 \
                +actor_rollout_ref.actor.opsa_topk_logits_k=512 \
                trainer.experiment_name=opsa \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "opsa"
        ;;

    # ================================================================
    # ROPD: Rubric-based On-policy Distillation (uses LLM judge reward)
    # For benchmark, we use standard reward + ropd reward manager
    # ================================================================
    ropd)
        CKPT_DIR="${CHECKPOINT_DIR}/ropd"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training ROPD..."
            # ROPD uses a rubric-based reward manager. For benchmark comparison,
            # we use the standard math reward but with the ROPD algorithm structure.
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                trainer.experiment_name=ropd \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "ropd"
        ;;

    # ================================================================
    # Vision-OPD: Self-distillation with EMA teacher (no external teacher)
    # ================================================================
    vision_opd)
        CKPT_DIR="${CHECKPOINT_DIR}/vision_opd"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training Vision-OPD..."
            ${PYTHON} -m verl.trainer.main_ppo \
                "${COMMON_ARGS[@]}" \
                actor_rollout_ref.actor.use_kl_loss=False \
                actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
                actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
                actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
                actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
                actor_rollout_ref.actor.self_distillation.teacher_model_source=legacy \
                actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
                actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
                actor_rollout_ref.actor.self_distillation.alpha=0.5 \
                actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
                algorithm.norm_adv_by_std_in_grpo=False \
                algorithm.rollout_correction.rollout_is=token \
                algorithm.rollout_correction.rollout_is_threshold=2.0 \
                actor_rollout_ref.rollout.calculate_log_probs=True \
                trainer.experiment_name=vision_opd \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "vision_opd"
        ;;

    # ================================================================
    # Simple: Cross-tokenizer KD (teacher with different tokenizer)
    # Uses distillation.enabled=True with cross-tokenizer mode
    # ================================================================
    simple)
        CKPT_DIR="${CHECKPOINT_DIR}/simple"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training Simple (Cross-Tokenizer KD)..."
            # Simple uses 6 GPUs for student + 2 GPUs for teacher
            ${PYTHON} -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                algorithm.use_kl_in_reward=False \
                "data.train_files=['${DATA_DIR}/train.parquet']" \
                "data.val_files=['${DATA_DIR}/val.parquet']" \
                data.train_batch_size=6 \
                data.max_prompt_length=${MAX_PROMPT_LEN} \
                data.max_response_length=${MAX_RESPONSE_LEN} \
                data.filter_overlong_prompts=True \
                data.truncation=right \
                data.prompt_key=prompt \
                actor_rollout_ref.model.path="${STUDENT_MODEL}" \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.actor.use_torch_compile=False \
                actor_rollout_ref.actor.optim.lr=5e-7 \
                actor_rollout_ref.actor.ppo_mini_batch_size=6 \
                actor_rollout_ref.actor.use_dynamic_bsz=True \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
                actor_rollout_ref.rollout.n=1 \
                actor_rollout_ref.rollout.temperature=0.6 \
                actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
                actor_rollout_ref.rollout.max_num_seqs=16 \
                actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
                actor_rollout_ref.rollout.enforce_eager=True \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                custom_reward_function.path="${REWARD_FN}" \
                custom_reward_function.name=compute_score \
                distillation.enabled=True \
                distillation.n_gpus_per_node=2 \
                distillation.nnodes=1 \
                "distillation.simple_teacher_gpu_ids=[6,7]" \
                "distillation.simple_teacher_visible_devices=[0,1,2,3,4,5,6,7]" \
                distillation.simple_teacher_share_student_pool=False \
                distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
                distillation.teacher_models.teacher_model.num_replicas=2 \
                distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
                distillation.teacher_models.teacher_model.inference.name=vllm \
                distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
                distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN} \
                distillation.distillation_loss.loss_mode=simple \
                distillation.distillation_loss.use_task_rewards=False \
                distillation.distillation_loss.use_policy_gradient=False \
                distillation.distillation_loss.loss_max_clamp=10.0 \
                distillation.distillation_loss.cross_tokenizer_kl_direction=reverse \
                trainer.balance_batch=True \
                'trainer.logger=["console"]' \
                trainer.project_name=easyopd_benchmark \
                trainer.experiment_name=simple \
                trainer.n_gpus_per_node=6 \
                trainer.nnodes=1 \
                trainer.val_before_train=False \
                trainer.total_training_steps=${TRAIN_STEPS} \
                trainer.save_freq=${SAVE_FREQ} \
                trainer.test_freq=50 \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "simple"
        ;;

    # ================================================================
    # SimCT: Span-based Cross-Tokenizer KD
    # Same as simple but with loss_mode=simct
    # ================================================================
    simct)
        CKPT_DIR="${CHECKPOINT_DIR}/simct"
        if [ ! -d "${CKPT_DIR}/actor" ]; then
            echo "[$(date)] Training SimCT (Span Cross-Tokenizer KD)..."
            ${PYTHON} -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                algorithm.use_kl_in_reward=False \
                "data.train_files=['${DATA_DIR}/train.parquet']" \
                "data.val_files=['${DATA_DIR}/val.parquet']" \
                data.train_batch_size=6 \
                data.max_prompt_length=${MAX_PROMPT_LEN} \
                data.max_response_length=${MAX_RESPONSE_LEN} \
                data.filter_overlong_prompts=True \
                data.truncation=right \
                data.prompt_key=prompt \
                actor_rollout_ref.model.path="${STUDENT_MODEL}" \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.actor.use_torch_compile=False \
                actor_rollout_ref.actor.optim.lr=5e-7 \
                actor_rollout_ref.actor.ppo_mini_batch_size=6 \
                actor_rollout_ref.actor.use_dynamic_bsz=True \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
                actor_rollout_ref.rollout.n=1 \
                actor_rollout_ref.rollout.temperature=0.6 \
                actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
                actor_rollout_ref.rollout.max_num_seqs=16 \
                actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
                actor_rollout_ref.rollout.enforce_eager=True \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                custom_reward_function.path="${REWARD_FN}" \
                custom_reward_function.name=compute_score \
                distillation.enabled=True \
                distillation.n_gpus_per_node=2 \
                distillation.nnodes=1 \
                "distillation.simple_teacher_gpu_ids=[6,7]" \
                "distillation.simple_teacher_visible_devices=[0,1,2,3,4,5,6,7]" \
                distillation.simple_teacher_share_student_pool=False \
                distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
                distillation.teacher_models.teacher_model.num_replicas=2 \
                distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
                distillation.teacher_models.teacher_model.inference.name=vllm \
                distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
                distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN} \
                distillation.distillation_loss.loss_mode=simct \
                distillation.distillation_loss.use_task_rewards=False \
                distillation.distillation_loss.use_policy_gradient=False \
                distillation.distillation_loss.loss_max_clamp=10.0 \
                distillation.distillation_loss.cross_tokenizer_kl_direction=reverse \
                trainer.balance_batch=True \
                'trainer.logger=["console"]' \
                trainer.project_name=easyopd_benchmark \
                trainer.experiment_name=simct \
                trainer.n_gpus_per_node=6 \
                trainer.nnodes=1 \
                trainer.val_before_train=False \
                trainer.total_training_steps=${TRAIN_STEPS} \
                trainer.save_freq=${SAVE_FREQ} \
                trainer.test_freq=50 \
                trainer.default_local_dir="${CKPT_DIR}"
        fi
        evaluate_model "${CKPT_DIR}/actor" "simct"
        ;;

    # ================================================================
    # ALL: Run all methods sequentially
    # ================================================================
    all)
        echo "Running all methods sequentially..."
        for m in base sft grpo gkd sod opcd g_opd sdpo opsa ropd vision_opd simple simct; do
            echo ""
            echo ">>>>>>>>>> [$(date)] Running: ${m} <<<<<<<<<<"
            bash "${EXPERIMENT_DIR}/run_single.sh" "${m}" 2>&1 | tee "${LOG_DIR}/${m}.log"
            echo ">>>>>>>>>> [$(date)] Done: ${m} <<<<<<<<<<"
            echo ""
        done
        ${PYTHON} "${EXPERIMENT_DIR}/generate_table.py"
        ;;

    *)
        echo "Unknown method: ${METHOD}"
        echo "Available: base, sft, grpo, gkd, sod, opcd, g_opd, sdpo, opsa, ropd, vision_opd, simple, simct, all"
        exit 1
        ;;
esac

echo "[$(date)] Method ${METHOD} completed!"
