#!/bin/bash
set -euo pipefail

# ============================================================
# Quick dry-run test for all 3 merged methods:
#   1. SOD (sod_new branch)
#   2. SDPO (self-distillation branch)
#   3. ALM (cross branch)
#
# Each method runs only 1-2 training steps to verify the code
# can initialize and run without errors on the merged codebase.
# ============================================================

EASYOPD_ROOT="/apdcephfs_cq8/share_1324356/qiyongzhong/SOD_Merge_Framework/EasyOPD_clone"
export PYTHONPATH="${EASYOPD_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1

PYTHON="/opt/conda/envs/OpenAgentRL/bin/python"
RAY="/opt/conda/envs/OpenAgentRL/bin/ray"
export PATH="/opt/conda/envs/OpenAgentRL/bin:${PATH}"

TEST_LOG_DIR="${EASYOPD_ROOT}/test_logs"
mkdir -p "${TEST_LOG_DIR}"

# Models available in our environment
STUDENT_SMALL="/apdcephfs_cq8/share_1324356/qiyongzhong/checkpoint/qwen3_1p7b_sft/global_step_115/huggingface"
TEACHER_4B="/apdcephfs_cq8/share_1324356/qiyongzhong/download_models/models--Gen-Verse--DemyAgent-4B/snapshots/6a097c80a5b60a106db46d9f72624988b078ad01"
QWEN3_8B="/apdcephfs_cq8/share_1324356/qiyongzhong/download_models/Qwen3-8B"

# Data
SOD_TRAIN_DATA="/apdcephfs_cq8/share_1324356/qiyongzhong/dataset/Gen-Verse/Open-AgentRL-30K/Open-AgentRL-30K.parquet"
SOD_VAL_DATA_1="/apdcephfs_cq8/share_1324356/qiyongzhong/dataset/Gen-Verse/Open-AgentRL-Eval/aime2025/aime_2025_problems.parquet"
SOD_VAL_DATA_2="/apdcephfs_cq8/share_1324356/qiyongzhong/dataset/Gen-Verse/Open-AgentRL-Eval/aime2024/aime_2024_problems.parquet"
SDPO_RAW_DATASET="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/dataset/mixed_math_code_10k"

SANDBOX_FUSION_URL="https://sd72ileknjkrkkoplocgg.apigateway-cn-beijing.volceapi.com/run_code?faasInstanceName=vefaas-d2xrxq4u-lottx7z2gg-d8p54qg31b6djm28q460-sandbox"
export SANDBOX_FUSION_URL

# Test output dirs
SOD_TEST_DIR="${EASYOPD_ROOT}/test_outputs/sod"
SDPO_TEST_DIR="${EASYOPD_ROOT}/test_outputs/sdpo"
ALM_TEST_DIR="${EASYOPD_ROOT}/test_outputs/alm"
mkdir -p "${SOD_TEST_DIR}" "${SDPO_TEST_DIR}" "${ALM_TEST_DIR}"

echo "============================================================"
echo "[$(date)] Starting merged codebase verification tests"
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

METHOD_TO_TEST="${1:-all}"

# ============================================================
# TEST 1: SOD (sod_new branch)
# ============================================================
run_sod_test() {
    echo ""
    echo "============================================================"
    echo "[$(date)] TEST 1: SOD (Step-wise On-policy Distillation)"
    echo "============================================================"

    cd "${EASYOPD_ROOT}"
    ${PYTHON} -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.002 \
        +algorithm.filter_groups.enable=False \
        +algorithm.filter_groups.metric=seq_reward \
        +algorithm.filter_groups.reward_std_threshold=0.0 \
        +algorithm.token_kl_reg.enable=True \
        +algorithm.token_kl_reg.gamma=1.0 \
        +algorithm.token_kl_reg.beta_min=0.0 \
        +algorithm.token_kl_reg.beta_max=0.10 \
        +algorithm.token_kl_reg.stepwise_enable=True \
        +algorithm.token_kl_reg.stepwise_epsilon=1e-6 \
        +algorithm.token_kl_reg.stepwise_delta=0.2 \
        +algorithm.token_kl_reg.stepwise_opd_coef=1.0 \
        "data.train_files=['${SOD_TRAIN_DATA}']" \
        "data.val_files=['${SOD_VAL_DATA_1}','${SOD_VAL_DATA_2}']" \
        data.return_raw_chat=True \
        data.train_batch_size=8 \
        data.max_prompt_length=2560 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.custom_cls.path=examples/sod/reward.py \
        data.custom_cls.name=CustomRLHFDataset \
        custom_reward_function.path=examples/sod/reward.py \
        custom_reward_function.name=compute_score \
        actor_rollout_ref.model.path=${STUDENT_SMALL} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        +actor_rollout_ref.ref.model.path=${TEACHER_4B} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/sod/sandbox_fusion_tool_config.yaml \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=4 \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=1024 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
        +reward_model.reward_kwargs.max_resp_len=4096 \
        "trainer.logger=['console']" \
        trainer.project_name=test_sod \
        trainer.experiment_name=test_sod_dryrun \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.log_val_generations=5 \
        trainer.save_freq=999 \
        trainer.default_local_dir=${SOD_TEST_DIR} \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        +trainer.total_training_steps=2 \
        2>&1 | tee "${TEST_LOG_DIR}/test_sod.log"

    if [ $? -eq 0 ]; then
        echo "[$(date)] ✅ SOD TEST PASSED"
    else
        echo "[$(date)] ❌ SOD TEST FAILED"
    fi
}

# ============================================================
# TEST 2: SDPO (Self-Distillation Policy Optimization)
# ============================================================
run_sdpo_test() {
    echo ""
    echo "============================================================"
    echo "[$(date)] TEST 2: SDPO (Self-Distillation Policy Optimization)"
    echo "============================================================"

    # First prepare a small RL prompt parquet
    SDPO_TRAIN_PARQUET="${SDPO_TEST_DIR}/rl_prompts_train.parquet"
    SDPO_VAL_PARQUET="${SDPO_TEST_DIR}/rl_prompts_val.parquet"

    if [ ! -f "${SDPO_TRAIN_PARQUET}" ]; then
        echo "[$(date)] Building SDPO test data..."
        ${PYTHON} - <<PYEOF
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

DATASET_DIR = "${SDPO_RAW_DATASET}"
TRAIN_PATH = "${SDPO_TRAIN_PARQUET}"
VAL_PATH = "${SDPO_VAL_PARQUET}"

print(f"Loading raw dataset {DATASET_DIR}")
ds = load_from_disk(DATASET_DIR)

records = []
for item in tqdm(list(ds)[:200], desc="building prompts"):
    msgs = item.get("messages") or []
    label = item.get("label", "")
    if not msgs:
        continue
    first_msg = msgs[0]
    user_msg = first_msg.get("content", "") if isinstance(first_msg, dict) else str(first_msg)
    data_source = "math" if any(k in user_msg.lower() for k in
        ["solve", "find", "calculate", "prove", "\\boxed"]) else "code"
    records.append({
        "prompt": [{"role": "user", "content": user_msg}],
        "data_source": data_source,
        "reward_model": {"ground_truth": "" if label is None else str(label)},
        "extra_info": {"index": len(records)},
    })

train = records[:100]
val   = records[100:120]
pd.DataFrame(train).to_parquet(TRAIN_PATH)
pd.DataFrame(val  ).to_parquet(VAL_PATH)
print(f"train: {len(train)} -> {TRAIN_PATH}")
print(f"val:   {len(val)} -> {VAL_PATH}")
PYEOF
    fi

    export VLLM_USE_V1=0
    REWARD_FN="${EASYOPD_ROOT}/experiments/_shared/scripts/feedback_reward_fn.py"

    cd /tmp
    ${PYTHON} -m verl.trainer.main_ppo \
        +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.norm_adv_by_std_in_grpo=False \
        +algorithm.rollout_correction.rollout_is=token \
        +algorithm.rollout_correction.rollout_is_threshold=2.0 \
        "data.train_files=['${SDPO_TRAIN_PARQUET}']" \
        "data.val_files=['${SDPO_VAL_PARQUET}']" \
        data.train_batch_size=8 \
        data.max_prompt_length=2048 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.shuffle=True \
        data.prompt_key=prompt \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="${QWEN3_8B}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_torch_compile=False \
        +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
        +actor_rollout_ref.actor.self_distillation.alpha=0.5 \
        +actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
        +actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
        +actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
        +actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
        +actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5 \
        +actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
        +actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
        +actor_rollout_ref.actor.self_distillation.max_reprompt_len=4096 \
        +actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
        +actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
        +actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=True \
        +actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.max_model_len=8192 \
        actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        custom_reward_function.path="${REWARD_FN}" \
        custom_reward_function.name=compute_score \
        trainer.balance_batch=True \
        'trainer.logger=["console"]' \
        trainer.project_name=test_sdpo \
        trainer.experiment_name=test_sdpo_dryrun \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.total_epochs=1 \
        trainer.save_freq=999 \
        trainer.test_freq=-1 \
        trainer.default_local_dir="${SDPO_TEST_DIR}" \
        +trainer.total_training_steps=2 \
        2>&1 | tee "${TEST_LOG_DIR}/test_sdpo.log"

    if [ $? -eq 0 ]; then
        echo "[$(date)] ✅ SDPO TEST PASSED"
    else
        echo "[$(date)] ❌ SDPO TEST FAILED"
    fi
    export VLLM_USE_V1=1
}

# ============================================================
# TEST 3: ALM (Approximate Likelihood Matching - Cross-Tokenizer)
# Note: Uses Qwen3-1.7B as student and Qwen3-8B as teacher
#       (original uses phi4-mini + Qwen2.5-7B which are not available)
# ============================================================
run_alm_test() {
    echo ""
    echo "============================================================"
    echo "[$(date)] TEST 3: ALM (Cross-Tokenizer OPD)"
    echo "============================================================"

    # Prepare ALM test data (reuse SDPO data format)
    ALM_TRAIN_PARQUET="${ALM_TEST_DIR}/rl_prompts_train.parquet"
    ALM_VAL_PARQUET="${ALM_TEST_DIR}/rl_prompts_val.parquet"

    if [ ! -f "${ALM_TRAIN_PARQUET}" ]; then
        echo "[$(date)] Building ALM test data..."
        ${PYTHON} - <<PYEOF
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

DATASET_DIR = "${SDPO_RAW_DATASET}"
TRAIN_PATH = "${ALM_TRAIN_PARQUET}"
VAL_PATH = "${ALM_VAL_PARQUET}"

print(f"Loading raw dataset {DATASET_DIR}")
ds = load_from_disk(DATASET_DIR)

records = []
for item in tqdm(list(ds)[:200], desc="building prompts"):
    msgs = item.get("messages") or []
    label = item.get("label", "")
    if not msgs:
        continue
    first_msg = msgs[0]
    user_msg = first_msg.get("content", "") if isinstance(first_msg, dict) else str(first_msg)
    data_source = "math" if any(k in user_msg.lower() for k in
        ["solve", "find", "calculate", "prove", "\\boxed"]) else "code"
    records.append({
        "prompt": [{"role": "user", "content": user_msg}],
        "data_source": data_source,
        "reward_model": {"ground_truth": "" if label is None else str(label)},
        "extra_info": {"index": len(records)},
    })

train = records[:100]
val   = records[100:120]
pd.DataFrame(train).to_parquet(TRAIN_PATH)
pd.DataFrame(val  ).to_parquet(VAL_PATH)
print(f"train: {len(train)} -> {TRAIN_PATH}")
print(f"val:   {len(val)} -> {VAL_PATH}")
PYEOF
    fi

    export VLLM_USE_V1=0
    REWARD_FN="${EASYOPD_ROOT}/experiments/_shared/scripts/reward_fn.py"

    cd /tmp
    ${PYTHON} -m verl.trainer.main_ppo \
        +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        "data.train_files=['${ALM_TRAIN_PARQUET}']" \
        "data.val_files=['${ALM_VAL_PARQUET}']" \
        data.train_batch_size=8 \
        data.max_prompt_length=2048 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.shuffle=True \
        data.prompt_key=prompt \
        actor_rollout_ref.model.path="${STUDENT_SMALL}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_torch_compile=False \
        +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.actor.policy_loss.loss_mode=alm \
        +actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0 \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.max_model_len=6145 \
        actor_rollout_ref.rollout.max_num_batched_tokens=6145 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
        custom_reward_function.path="${REWARD_FN}" \
        custom_reward_function.name=compute_score \
        +distillation.enabled=True \
        +distillation.n_gpus_per_node=8 \
        +distillation.nnodes=1 \
        "+distillation.simple_teacher_gpu_ids=[0,1,2,3,4,5,6,7]" \
        "+distillation.simple_teacher_visible_devices=[0,1,2,3,4,5,6,7]" \
        +distillation.simple_teacher_share_student_pool=True \
        +distillation.simple_teacher_num_gpus_per_actor=0 \
        +distillation.teacher_models.teacher_model.model_path="${QWEN3_8B}" \
        +distillation.teacher_models.teacher_model.num_replicas=8 \
        +distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
        +distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size=1 \
        +distillation.teacher_models.teacher_model.inference.name=sglang \
        +distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5 \
        +distillation.teacher_models.teacher_model.inference.max_model_len=6145 \
        +distillation.distillation_loss.loss_mode=alm \
        +distillation.distillation_loss.use_cross_tokenizer=True \
        +distillation.distillation_loss.use_task_rewards=False \
        +distillation.distillation_loss.use_policy_gradient=False \
        +distillation.distillation_loss.distillation_loss_coef=1.0 \
        +distillation.distillation_loss.loss_max_clamp=10.0 \
        +distillation.distillation_loss.cross_tokenizer_kl_direction=reverse \
        +distillation.distillation_loss.alm_temperature=100.0 \
        +distillation.distillation_loss.alm_f_divergence=kl \
        +distillation.distillation_loss.alm_debiasing=false \
        trainer.balance_batch=True \
        'trainer.logger=["console"]' \
        trainer.project_name=test_alm \
        trainer.experiment_name=test_alm_dryrun \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.total_epochs=1 \
        trainer.save_freq=999 \
        trainer.test_freq=-1 \
        trainer.default_local_dir="${ALM_TEST_DIR}" \
        +trainer.total_training_steps=2 \
        2>&1 | tee "${TEST_LOG_DIR}/test_alm.log"

    if [ $? -eq 0 ]; then
        echo "[$(date)] ✅ ALM TEST PASSED"
    else
        echo "[$(date)] ❌ ALM TEST FAILED"
    fi
    export VLLM_USE_V1=1
}

# ============================================================
# TEST 4: SimCT (Span-based Cross-Tokenizer KD)
# Note: Uses Qwen3-1.7B as student and Qwen3-8B as teacher
#       Teacher backend falls back to HFTeacherEngine (no sglang)
# ============================================================
run_simct_test() {
    echo ""
    echo "============================================================"
    echo "[$(date)] TEST 4: SimCT (Cross-Tokenizer Span-based KD)"
    echo "============================================================"

    # Reuse ALM test data (same format)
    SIMCT_TEST_DIR="${EASYOPD_ROOT}/test_outputs/simct"
    mkdir -p "${SIMCT_TEST_DIR}"
    SIMCT_TRAIN_PARQUET="${SIMCT_TEST_DIR}/rl_prompts_train.parquet"
    SIMCT_VAL_PARQUET="${SIMCT_TEST_DIR}/rl_prompts_val.parquet"

    if [ ! -f "${SIMCT_TRAIN_PARQUET}" ]; then
        echo "[$(date)] Building SimCT test data..."
        ${PYTHON} - <<PYEOF
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

DATASET_DIR = "${SDPO_RAW_DATASET}"
TRAIN_PATH = "${SIMCT_TRAIN_PARQUET}"
VAL_PATH = "${SIMCT_VAL_PARQUET}"

print(f"Loading raw dataset {DATASET_DIR}")
ds = load_from_disk(DATASET_DIR)

records = []
for item in tqdm(list(ds)[:200], desc="building prompts"):
    msgs = item.get("messages") or []
    label = item.get("label", "")
    if not msgs:
        continue
    first_msg = msgs[0]
    user_msg = first_msg.get("content", "") if isinstance(first_msg, dict) else str(first_msg)
    data_source = "math" if any(k in user_msg.lower() for k in
        ["solve", "find", "calculate", "prove", "\\boxed"]) else "code"
    records.append({
        "prompt": [{"role": "user", "content": user_msg}],
        "data_source": data_source,
        "reward_model": {"ground_truth": "" if label is None else str(label)},
        "extra_info": {"index": len(records)},
    })

train = records[:100]
val   = records[100:120]
pd.DataFrame(train).to_parquet(TRAIN_PATH)
pd.DataFrame(val  ).to_parquet(VAL_PATH)
print(f"train: {len(train)} -> {TRAIN_PATH}")
print(f"val:   {len(val)} -> {VAL_PATH}")
PYEOF
    fi

    export VLLM_USE_V1=0
    REWARD_FN="${EASYOPD_ROOT}/experiments/_shared/scripts/reward_fn.py"

    cd /tmp
    ${PYTHON} -m verl.trainer.main_ppo \
        +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        "data.train_files=['${SIMCT_TRAIN_PARQUET}']" \
        "data.val_files=['${SIMCT_VAL_PARQUET}']" \
        data.train_batch_size=8 \
        data.max_prompt_length=2048 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.shuffle=True \
        data.prompt_key=prompt \
        actor_rollout_ref.model.path="${STUDENT_SMALL}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_torch_compile=False \
        +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.actor.policy_loss.loss_mode=simct \
        +actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0 \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.max_model_len=6145 \
        actor_rollout_ref.rollout.max_num_batched_tokens=6145 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
        custom_reward_function.path="${REWARD_FN}" \
        custom_reward_function.name=compute_score \
        +distillation.enabled=True \
        +distillation.n_gpus_per_node=8 \
        +distillation.nnodes=1 \
        "+distillation.simple_teacher_gpu_ids=[0,1,2,3,4,5,6,7]" \
        "+distillation.simple_teacher_visible_devices=[0,1,2,3,4,5,6,7]" \
        +distillation.simple_teacher_share_student_pool=True \
        +distillation.simple_teacher_num_gpus_per_actor=0 \
        +distillation.teacher_models.teacher_model.model_path="${QWEN3_8B}" \
        +distillation.teacher_models.teacher_model.num_replicas=8 \
        +distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
        +distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size=1 \
        +distillation.teacher_models.teacher_model.inference.name=sglang \
        +distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5 \
        +distillation.teacher_models.teacher_model.inference.max_model_len=6145 \
        +distillation.distillation_loss.loss_mode=simct \
        +distillation.distillation_loss.use_cross_tokenizer=True \
        +distillation.distillation_loss.use_task_rewards=False \
        +distillation.distillation_loss.use_policy_gradient=False \
        +distillation.distillation_loss.distillation_loss_coef=1.0 \
        +distillation.distillation_loss.loss_max_clamp=10.0 \
        +distillation.distillation_loss.cross_tokenizer_kl_direction=reverse \
        trainer.balance_batch=True \
        'trainer.logger=["console"]' \
        trainer.project_name=test_simct \
        trainer.experiment_name=test_simct_dryrun \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.total_epochs=1 \
        trainer.save_freq=999 \
        trainer.test_freq=-1 \
        trainer.default_local_dir="${SIMCT_TEST_DIR}" \
        +trainer.total_training_steps=2 \
        2>&1 | tee "${TEST_LOG_DIR}/test_simct.log"

    if [ $? -eq 0 ]; then
        echo "[$(date)] ✅ SimCT TEST PASSED"
    else
        echo "[$(date)] ❌ SimCT TEST FAILED"
    fi
    export VLLM_USE_V1=1
}

# ============================================================
# Run tests based on argument
# ============================================================
case "${METHOD_TO_TEST}" in
    sod)
        run_sod_test
        ;;
    sdpo)
        run_sdpo_test
        ;;
    alm|cross)
        run_alm_test
        ;;
    simct)
        run_simct_test
        ;;
    all)
        run_sod_test
        run_sdpo_test
        run_alm_test
        run_simct_test
        ;;
    *)
        echo "Usage: $0 [sod|sdpo|alm|simct|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "[$(date)] All requested tests completed."
echo "Logs: ${TEST_LOG_DIR}/"
echo "============================================================"
