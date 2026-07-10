#!/bin/bash
set -euo pipefail

# ============================================================
# SimCT (Span-based Cross-Tokenizer KD) Test
# Following EasyOPD examples/simct/run_simct.sh
#
# Student: Phi-4-mini-instruct (SFT-warmed checkpoint-40)
# Teacher: Qwen2.5-7B-Instruct
# This is a TRUE cross-tokenizer scenario (different tokenizers)
#
# Teacher backend: HFTeacherEngine (fallback, no sglang installed)
# Student rollout: vLLM
# ============================================================

EASYOPD_ROOT="/apdcephfs_cq8/share_1324356/qiyongzhong/SOD_Merge_Framework/EasyOPD_clone"
export PYTHONPATH="${EASYOPD_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export EASYOPD_SIMCT_DEBUG=1
export NCCL_TIMEOUT=1800000

PYTHON="/opt/conda/envs/OpenAgentRL/bin/python"
RAY="/opt/conda/envs/OpenAgentRL/bin/ray"
export PATH="/opt/conda/envs/OpenAgentRL/bin:${PATH}"

# ===== Models =====
# Student: SFT-warmed phi4-mini (KDFlow checkpoint-40)
STUDENT_MODEL="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/KDFlow/output/ckpts/phi4-mini-sft-warmup-10k-qwen-lr2e-6/checkpoint-40"
# Teacher: Qwen2.5-7B-Instruct (different tokenizer from phi4-mini!)
TEACHER_MODEL="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/models/Qwen2.5-7B-Instruct"

# ===== Data =====
DATA_DIR="${EASYOPD_ROOT}/test_outputs/simct_data"
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
VAL_PARQUET="${DATA_DIR}/test.parquet"

# ===== Output =====
SIMCT_TEST_DIR="${EASYOPD_ROOT}/test_outputs/simct_cross_tok"
mkdir -p "${SIMCT_TEST_DIR}"

# ===== Reward function (neutral placeholder for KD-only training) =====
REWARD_FN="${EASYOPD_ROOT}/examples/simct/reward.py"

echo "============================================================"
echo "[$(date)] SimCT Cross-Tokenizer Test"
echo "  Student: ${STUDENT_MODEL}"
echo "  Teacher: ${TEACHER_MODEL}"
echo "  Data:    ${TRAIN_PARQUET}"
echo "============================================================"

# ===== Install EasyOPD =====
echo "[$(date)] Installing EasyOPD..."
cd "${EASYOPD_ROOT}"
${PYTHON} -m pip install -e . --quiet 2>&1 | tail -3

# ===== Start Ray =====
echo "[$(date)] Starting Ray cluster..."
${RAY} stop --force 2>/dev/null || true
sleep 3
unset RAY_ADDRESS
${RAY} start --head --disable-usage-stats --num-cpus=32 --num-gpus=8 --include-dashboard=false
export RAY_ADDRESS="$(hostname -I | awk '{print $1}'):6379"
sleep 5

# Verify Ray
${PYTHON} -c "
import ray
ray.init(address='${RAY_ADDRESS}', ignore_reinit_error=True, log_to_driver=False)
@ray.remote
def ping(): return 'ok'
assert ray.get(ping.remote(), timeout=10) == 'ok'
print('[RayCheck] OK')
ray.shutdown()
"

echo "[$(date)] Running SimCT training (2 steps dry-run)..."

# Following examples/simct/run_simct.sh layout:
# - 6 GPUs for student (FSDP/vLLM), 2 GPUs for teacher (HFTeacherEngine)
# - Teacher uses dedicated GPUs 6,7 (NOT shared with student)
# - Batch size = 6 (divisible by 6 student GPUs)
# - Short sequences for fast testing: max_prompt=512, max_response=512
cd /tmp
${PYTHON} -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "data.train_files=['${TRAIN_PARQUET}']" \
    "data.val_files=['${VAL_PARQUET}']" \
    data.train_batch_size=6 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=simct \
    +actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.max_model_len=1025 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name=compute_score \
    +distillation.enabled=True \
    +distillation.n_gpus_per_node=2 \
    +distillation.nnodes=1 \
    "+distillation.simple_teacher_gpu_ids=[6,7]" \
    "+distillation.simple_teacher_visible_devices=[0,1,2,3,4,5,6,7]" \
    +distillation.simple_teacher_share_student_pool=False \
    +distillation.simple_teacher_num_gpus_per_actor=1.0 \
    +distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    +distillation.teacher_models.teacher_model.num_replicas=2 \
    +distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    +distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size=1 \
    +distillation.teacher_models.teacher_model.inference.name=sglang \
    +distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5 \
    +distillation.teacher_models.teacher_model.inference.max_model_len=1025 \
    +distillation.distillation_loss.loss_mode=simct \
    +distillation.distillation_loss.use_cross_tokenizer=True \
    +distillation.distillation_loss.use_task_rewards=False \
    +distillation.distillation_loss.use_policy_gradient=False \
    +distillation.distillation_loss.distillation_loss_coef=1.0 \
    +distillation.distillation_loss.loss_max_clamp=10.0 \
    +distillation.distillation_loss.cross_tokenizer_kl_direction=reverse \
    trainer.balance_batch=True \
    'trainer.logger=["console"]' \
    trainer.project_name=test_simct_cross_tok \
    trainer.experiment_name=phi4mini_qwen25_simct_dryrun \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.save_freq=999 \
    trainer.test_freq=-1 \
    trainer.default_local_dir="${SIMCT_TEST_DIR}" \
    +trainer.total_training_steps=2 \
    2>&1 | tee "${EASYOPD_ROOT}/test_logs/test_simct_cross_tok.log"

if [ $? -eq 0 ]; then
    echo "[$(date)] ✅ SimCT Cross-Tokenizer TEST PASSED"
else
    echo "[$(date)] ❌ SimCT Cross-Tokenizer TEST FAILED"
fi

export VLLM_USE_V1=1
echo "[$(date)] Done."
