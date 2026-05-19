#!/usr/bin/env bash
# EasyOPD `simple` — cross-tokenizer on-policy distillation launch script.
#
# Mirrors the KDFlow reference run
#   scripts/ctopd_phi4/qwen25_phi4_simple_mix10k_lr5e-7.sh
# (Teacher: Qwen2.5-7B-Instruct  ->  Student: phi-4-mini SFT warmup,
#  Algorithm: SimpleCrossTokenizerKD, Mixed Math+Code 10K, lr=5e-7)
# but routes through verl's standard `verl.trainer.main_ppo` entry point with
# `distillation.distillation_loss.loss_mode=simple` so the EasyOPD-side
# loss (registered via `easyopd.methods.simple.losses.register_simple_loss`)
# is dispatched.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- shared paths (KDFlow-aligned) ----
LOCAL_STUDENT_DIR=${LOCAL_STUDENT_DIR:-$HOME/workspace/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6}
REMOTE_STUDENT_DIR=${REMOTE_STUDENT_DIR:-/apdcephfs_cq8/share_1324356/shinejiesun/workspace/KDFlow/output/ckpts/phi4-mini-sft-warmup-10k-qwen-lr2e-6}
LOCAL_TEACHER_DIR=${LOCAL_TEACHER_DIR:-$HOME/workspace/models/Qwen2.5-7B-Instruct}
REMOTE_TEACHER_DIR=${REMOTE_TEACHER_DIR:-/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-7B-Instruct}

REMOTE_DATASET_DIR=${REMOTE_DATASET_DIR:-/apdcephfs_cq8/share_1324356/shinejiesun/workspace/dataset/mixed_math_code_10k}
LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-$HOME/data/mixed_math_code_10k}

# ---- sync student checkpoint to local SSD (KDFlow parity) ----
if [ ! -d "${LOCAL_STUDENT_DIR}/checkpoint-40" ]; then
    echo "[run_simple] syncing student checkpoint to local SSD..."
    mkdir -p "${LOCAL_STUDENT_DIR}"
    rsync -ah --info=progress2 \
        "${REMOTE_STUDENT_DIR}/checkpoint-40/" \
        "${LOCAL_STUDENT_DIR}/checkpoint-40/"
fi

# ---- sync teacher model to local SSD ----
if [ ! -f "${LOCAL_TEACHER_DIR}/config.json" ]; then
    echo "[run_simple] syncing teacher model to local SSD..."
    mkdir -p "$(dirname "${LOCAL_TEACHER_DIR}")"
    rsync -ah --info=progress2 \
        "${REMOTE_TEACHER_DIR}/" \
        "${LOCAL_TEACHER_DIR}/"
fi

# ---- prepare verl-style parquet dataset ----
if [ ! -f "${LOCAL_DATA_DIR}/train.parquet" ]; then
    echo "[run_simple] converting ${REMOTE_DATASET_DIR} -> parquet..."
    python3 "${SCRIPT_DIR}/prepare_data.py" \
        --src "${REMOTE_DATASET_DIR}" \
        --dst "${LOCAL_DATA_DIR}"
fi

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-${LOCAL_STUDENT_DIR}/checkpoint-40}
TEACHER_MODEL=${TEACHER_MODEL:-${LOCAL_TEACHER_DIR}}

NNODES=${NNODES:-1}
# On an 8-GPU node, verl accounts the student actor/rollout pool and the
# EasyOPD teacher sidecar pool separately. Keep the defaults within the
# physical node: 6 GPUs for student actor/rollout + 2 GPUs for teacher.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}

# `simple` is a cross-tokenizer KD mode. We bypass top-k / response_length=1
# rewrites in the teacher rollout config (handled by verl, gated on
# DistillationLossSettings.use_cross_tokenizer).
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-simple}
use_policy_gradient=${USE_POLICY_GRADIENT:-False}
# KDFlow uses --kd_loss_fn rkl => reverse KL.
kl_direction=${KL_DIRECTION:-reverse}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
# KDFlow: --max_len 8192, --generate_max_len 4096 (prompt budget ~= 4096).
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

# KDFlow: --learning_rate 5e-7.
actor_lr=${ACTOR_LR:-5e-7}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.6}
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.6}

# KDFlow: --num_epochs 1.
total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-50}

project_name=${PROJECT_NAME:-easyopd_simple_xtok}
experiment_name=${EXPERIMENT_NAME:-qwen25_phi4_simple_mix10k_lr5e-7}
# ---- end user-adjustable ----

train_files=${TRAIN_FILES:-"['${LOCAL_DATA_DIR}/train.parquet']"}
val_files=${VAL_FILES:-"['${LOCAL_DATA_DIR}/test.parquet']"}

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# Note: `simple` runs the teacher as an independent HF forward worker.
# We still declare `teacher_models.teacher_model.model_path` so the
# DistillationConfig accepts the YAML, but the inference engine name is
# bypassed for cross-tokenizer mode (validate_and_prepare_for_distillation
# returns early when use_cross_tokenizer=True).
EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.cross_tokenizer_kl_direction=${kl_direction}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
