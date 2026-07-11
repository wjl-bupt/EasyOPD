#!/bin/bash
set -euo pipefail
trap 'rc=$?; echo "[FATAL] launch.sh exited with code $rc at line $LINENO (last cmd: $BASH_COMMAND)" >&2' ERR

# ============================================================
# G-OPD: Generalized On-Policy Distillation with Reward Extrapolation
# Paper: https://arxiv.org/abs/2602.12125
#
# G-OPD introduces a reward scaling factor (lambda) and a flexible reference
# model to generalize standard OPD. Building on G-OPD, ExOPD (On-Policy
# Distillation with Reward Extrapolation) outperforms standard OPD.
#
# Key features:
#   - Reward Scaling (lambda): Controls teacher signal strength
#     lambda=1.0: standard OPD, lambda>1.0: ExOPD (recommended: 1.25)
#   - Uses reverse KL as advantages (on-policy distillation)
#   - Base model normalization for reward extrapolation
#   - Supports multi-teacher distillation and context distillation
#
# Pipeline:
#   1. Prepare RL prompt parquet (train.parquet / val.parquet)
#   2. (Re)start Ray
#   3. GRPO + G-OPD (reverse KL advantages + reward extrapolation)
#      via verl.trainer.main_ppo with only_reverse_kl_advantages=True
#   4. Merge each global_step_X/actor/ -> HF format
#   5. Evaluate every merged ckpt on math500 + gsm8k
#
# G-OPD modifies the advantage computation in dp_actor.py:
#   - Standard OPD (lambda=1.0): A = -(old_log_probs - ref_log_prob)
#   - ExOPD (lambda=1.25): A = -[(old - base) - lambda*(ref - base)]
# ============================================================

export PYTHONPATH="/path/to/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

unset RAY_ADDRESS

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
RAY="/opt/conda/envs/OpenAgentRL-sj/bin/ray"
export PATH="/opt/conda/envs/OpenAgentRL-sj/bin:${PATH}"

# Single-node Ray settings
RAY_NODE_IP="${RAY_NODE_IP:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_HEAD_ADDRESS="${RAY_NODE_IP}:${RAY_PORT}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
RAY_READY_RETRIES="${RAY_READY_RETRIES:-12}"

# ----------------- Paths -----------------
EASYOPD_ROOT="/path/to/EasyOPD"
EXPERIMENT_DIR="${EASYOPD_ROOT}/experiments"
EXP_DIR="${EXPERIMENT_DIR}/06_general_opd"
SHARED_SCRIPTS="${EXPERIMENT_DIR}/_shared/scripts"

METHOD_DIR="${EXP_DIR}/methods/g_opd"
RESULTS_DIR="${METHOD_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Student = Qwen2.5-1.5B-Instruct (same-tokenizer OPD starting point)
STUDENT_MODEL="/path/to/workspace/workspace/models/Qwen2.5-1.5B-Instruct"
# Teacher = Qwen2.5-7B-Instruct (same tokenizer, larger model)
TEACHER_MODEL="/path/to/workspace/workspace/models/Qwen2.5-7B-Instruct"
# Base model = Student's initial state (for reward normalization in ExOPD)
BASE_MODEL="/path/to/workspace/workspace/models/Qwen2.5-1.5B-Instruct"

# RL prompt data
TRAIN_DATA_DIR="${EXP_DIR}/train_data"
RL_TRAIN_PARQUET="${TRAIN_DATA_DIR}/rl_prompts_train.parquet"
RL_VAL_PARQUET="${TRAIN_DATA_DIR}/rl_prompts_val.parquet"

REWARD_FN="${SHARED_SCRIPTS}/reward_fn.py"

# ----------------- Run dir layout -----------------
EXP_NAME="06_general_opd"
METHOD="g_opd"
RUN_NAME="g_opd_qwen25_1.5b"

RUNS_ROOT="/path/to/models/runs"
RUN_DIR="${RUNS_ROOT}/${EXP_NAME}/${METHOD}/${RUN_NAME}"
FSDP_CKPT_DIR="${RUN_DIR}/fsdp"
HF_CKPT_DIR="${RUN_DIR}/hf"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${FSDP_CKPT_DIR}" "${HF_CKPT_DIR}" "${LOG_DIR}"

# ----------------- Hyperparameters -----------------
N_GPUS=8

MAX_PROMPT_LEN=2048
MAX_RESPONSE_LEN=4096
MAX_MODEL_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1 ))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

ACTOR_LR=1e-5
LR_WARMUP_RATIO=0.1
TOTAL_EPOCHS=2

TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=64
EXPECTED_FINAL_STEP=308

SAVE_FREQ=77
TEST_FREQ=-1

ROLLOUT_TP=1
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_GPU_MEM_UTIL=0.55
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}

# Speed/Memory tier
ENABLE_GRAD_CKPT=False
FSDP_PARAM_OFFLOAD=True
FSDP_OPTIMIZER_OFFLOAD=True

# G-OPD-specific parameters
LAMBDA_VALS=1.25              # Reward scaling factor (1.25 = ExOPD, recommended)
ONLY_REVERSE_KL_ADV=True     # Use reverse KL as advantages (core G-OPD mechanism)
MULTI_TEACHER_DISTILL=False   # Single-teacher mode

echo "[$(date)] G-OPD: lambda=${LAMBDA_VALS}, reverse_kl_adv=${ONLY_REVERSE_KL_ADV}, multi_teacher=${MULTI_TEACHER_DISTILL}"
echo "[$(date)] G-OPD: grad_ckpt=${ENABLE_GRAD_CKPT}, rollout_mem=${ROLLOUT_GPU_MEM_UTIL}"

# ============================================================
# Step 1: Prepare RL prompt parquet (train + val)
# ============================================================
REBUILD_RL_PARQUET=0
if [ ! -f "${RL_TRAIN_PARQUET}" ] || [ ! -f "${RL_VAL_PARQUET}" ]; then
    REBUILD_RL_PARQUET=1
else
    if ${PYTHON} - <<PYEOF
import sys
import pandas as pd

paths = ["${RL_TRAIN_PARQUET}", "${RL_VAL_PARQUET}"]

def normalize_prompt(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value

def prompt_schema_ok(value):
    value = normalize_prompt(value)
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if hasattr(first, "as_py"):
        first = first.as_py()
    return isinstance(first, dict) and "role" in first and "content" in first

for path in paths:
    df = pd.read_parquet(path, columns=["prompt"])
    if len(df) == 0 or not prompt_schema_ok(df.iloc[0]["prompt"]):
        got = type(df.iloc[0]["prompt"]).__name__ if len(df) else "empty"
        print(f"[Step1] Existing parquet has invalid prompt schema: {path} (got {got})", file=sys.stderr)
        raise SystemExit(1)

print("[Step1] Existing RL prompt parquet schema looks valid.")
PYEOF
    then
        REBUILD_RL_PARQUET=0
    else
        REBUILD_RL_PARQUET=1
    fi
fi

if [ "${REBUILD_RL_PARQUET}" = "1" ]; then
    echo "[$(date)] ===== Step 1: Building RL prompt parquet ====="
    mkdir -p "${TRAIN_DATA_DIR}"
    ${PYTHON} - <<PYEOF
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

DATASET_DIR = "/path/to/workspace/workspace/dataset/mixed_math_code_10k"
TRAIN_PATH = "${RL_TRAIN_PARQUET}"
VAL_PATH = "${RL_VAL_PARQUET}"

print(f"[Step1] Loading raw dataset {DATASET_DIR}")
ds = load_from_disk(DATASET_DIR)

records = []
for item in tqdm(ds, desc="[Step1] building prompts"):
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

train = records[:9900]
val   = records[9900:9950]
pd.DataFrame(train).to_parquet(TRAIN_PATH)
pd.DataFrame(val  ).to_parquet(VAL_PATH)
print(f"[Step1] train: {len(train)} -> {TRAIN_PATH}")
print(f"[Step1] val:   {len(val)} -> {VAL_PATH}")
PYEOF
    echo "[$(date)] ===== Step 1 Done ====="
else
    echo "[$(date)] [Step 1] RL prompt parquet exists with valid schema, skipping."
fi

# ============================================================
# Step 2: (Re)start Ray on this node
# ============================================================
if [ "${SKIP_RAY_RESTART:-0}" != "1" ]; then
    echo "[$(date)] ===== Step 2: Restarting Ray ====="
    unset RAY_ADDRESS
    ${RAY} stop --force 2>/dev/null || true
    sleep 5
    # Thorough cleanup of Ray residual state
    rm -rf /tmp/ray/session_* /tmp/ray/ray_current_cluster /tmp/ray/*.json 2>/dev/null || true
    # Kill any orphaned Ray processes that ray stop might have missed
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    sleep 2
    ${RAY} start --head --disable-usage-stats --node-ip-address=${RAY_NODE_IP} --port=${RAY_PORT} --num-cpus=${RAY_NUM_CPUS} --num-gpus=${N_GPUS} --include-dashboard=false
fi

export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
echo "[$(date)] Ray address: ${RAY_ADDRESS} (num_cpus=${RAY_NUM_CPUS}, num_gpus=${N_GPUS})"

ray_ready=0
for attempt in $(seq 1 ${RAY_READY_RETRIES}); do
    if timeout 30 ${PYTHON} - <<'PYEOF'
import os
import ray

ray_address = os.environ["RAY_ADDRESS"]
ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)

@ray.remote
def ping():
    return "ok"

assert ray.get(ping.remote(), timeout=10) == "ok"
print(f"[RayCheck] ray.init({ray_address}) + remote ping ok")
os._exit(0)
PYEOF
    then
        ray_ready=1
        break
    fi
    echo "[$(date)] [RayCheck] attempt ${attempt}/${RAY_READY_RETRIES} failed; retrying..."
    sleep 5
done

if [ "${ray_ready}" != "1" ]; then
    echo "[$(date)] ERROR: Ray did not pass remote ping health check."
    exit 1
fi

# ============================================================
# Step 3: GRPO + G-OPD Training (Reverse KL Advantages + Reward Extrapolation)
# ============================================================
LARGEST_EXISTING_STEP=0
shopt -s nullglob
for _ckpt in "${FSDP_CKPT_DIR}"/global_step_*; do
    _step=${_ckpt##*/global_step_}
    if [[ "${_step}" =~ ^[0-9]+$ ]] && (( _step > LARGEST_EXISTING_STEP )); then
        LARGEST_EXISTING_STEP=${_step}
    fi
done
shopt -u nullglob

if [ "${FORCE_RETRAIN:-0}" = "1" ]; then
    echo "[$(date)] FORCE_RETRAIN=1, will retrain even if checkpoints exist."
    SKIP_TRAIN=0
elif [ "${LARGEST_EXISTING_STEP}" -ge "${EXPECTED_FINAL_STEP}" ]; then
    echo "[$(date)] Found existing global_step_${LARGEST_EXISTING_STEP} (>= expected final ${EXPECTED_FINAL_STEP}); skipping Step 3."
    SKIP_TRAIN=1
else
    echo "[$(date)] Largest existing step = ${LARGEST_EXISTING_STEP} < expected ${EXPECTED_FINAL_STEP}; will (re)run training."
    SKIP_TRAIN=0
fi

if [ "${SKIP_TRAIN}" = "0" ]; then
echo "[$(date)] ===== Step 3: GRPO + G-OPD Training (ExOPD lambda=${LAMBDA_VALS}) ====="
echo "Student:        ${STUDENT_MODEL}"
echo "Teacher:        ${TEACHER_MODEL}"
echo "Base model:     ${BASE_MODEL}"
echo "Train parquet:  ${RL_TRAIN_PARQUET}"
echo "Val parquet:    ${RL_VAL_PARQUET}"
echo "batch=${TRAIN_BATCH_SIZE}, mini=${PPO_MINI_BATCH_SIZE}, lr=${ACTOR_LR}, epochs=${TOTAL_EPOCHS}"
echo "G-OPD: lambda=${LAMBDA_VALS}, reverse_kl_adv=${ONLY_REVERSE_KL_ADV}"

TRAIN_LAUNCH_CWD="/tmp/easyopd_${EXP_NAME}_${METHOD}_${RUN_NAME}"
mkdir -p "${TRAIN_LAUNCH_CWD}"

(
cd "${TRAIN_LAUNCH_CWD}"
${PYTHON} -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "data.train_files=['${RL_TRAIN_PARQUET}']" \
    "data.val_files=['${RL_VAL_PARQUET}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRAD_CKPT} \
    actor_rollout_ref.actor.use_torch_compile=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=${ONLY_REVERSE_KL_ADV} \
    actor_rollout_ref.actor.policy_loss.lambda_vals=${LAMBDA_VALS} \
    actor_rollout_ref.actor.policy_loss.multi_teacher_distill=${MULTI_TEACHER_DISTILL} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_RATIO} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
    +actor_rollout_ref.model.base_model_path="${BASE_MODEL}" \
    +actor_rollout_ref.ref.model.base_model_path="${BASE_MODEL}" \
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    'trainer.logger=["console"]' \
    trainer.project_name=easyopd-g-opd \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default_local_dir="${FSDP_CKPT_DIR}" \
    2>&1
) | tee "${LOG_DIR}/train.log"

echo "[$(date)] ===== Step 3: Training Completed ====="
fi

# ============================================================
# Step 4: Merge each global_step_X/actor/ -> HF format
# ============================================================
${RAY} stop --force 2>/dev/null || true

shopt -s nullglob
_ALL_CKPTS_RAW=( "${FSDP_CKPT_DIR}"/global_step_* )
shopt -u nullglob
ALL_CKPTS=()
if [ ${#_ALL_CKPTS_RAW[@]} -gt 0 ]; then
    while IFS= read -r _line; do
        ALL_CKPTS+=( "${_line}" )
    done < <(printf '%s\n' "${_ALL_CKPTS_RAW[@]}" | awk -F'global_step_' '{print $NF"\t"$0}' | sort -n -k1,1 | cut -f2-)
fi
if [ ${#ALL_CKPTS[@]} -eq 0 ]; then
    echo "[$(date)] ERROR: No checkpoint found in ${FSDP_CKPT_DIR}/global_step_*"
    exit 1
fi

echo "[$(date)] ===== Step 4: Merging ${#ALL_CKPTS[@]} actor checkpoint(s) to HF format ====="
for CKPT_DIR in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT_DIR}")
    ACTOR_DIR="${CKPT_DIR}/actor"
    TARGET_DIR="${HF_CKPT_DIR}/${STEP_NAME}"

    if [ ! -d "${ACTOR_DIR}" ]; then
        echo "[$(date)] [${STEP_NAME}] No actor/ subdir, skipping."
        continue
    fi
    if [ -f "${TARGET_DIR}/model.safetensors" ] || [ -f "${TARGET_DIR}/pytorch_model.bin" ]; then
        echo "[$(date)] [${STEP_NAME}] Already merged at ${TARGET_DIR}, skipping."
        continue
    fi

    echo "[$(date)] [${STEP_NAME}] Merging ${ACTOR_DIR} -> ${TARGET_DIR}"
    ${PYTHON} ${SHARED_SCRIPTS}/merge_fsdp.py \
        --ckpt_dir "${ACTOR_DIR}" \
        --base_model "${STUDENT_MODEL}" \
        --output_dir "${TARGET_DIR}" \
        2>&1 | tee "${LOG_DIR}/merge_${STEP_NAME}.log"
done
echo "[$(date)] ===== Step 4: Merge Completed ====="

# ============================================================
# Step 5: Evaluate every merged checkpoint
# ============================================================
echo "[$(date)] ===== Step 5: Evaluating G-OPD checkpoint(s) ====="

eval_already_done() {
    local details_json="$1"
    [ -f "${details_json}" ] || return 1
    local total
    total=$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('total',0))" "${details_json}" 2>/dev/null || echo 0)
    [ "${total}" -gt 0 ]
}

run_eval_one() {
    local merged_dir="$1" tag="$2" bench="$3" step_name="$4"
    local details_json="${RESULTS_DIR}/${tag}_${bench}_details.json"

    if [ "${FORCE_REEVAL:-0}" != "1" ] && eval_already_done "${details_json}"; then
        echo "[$(date)] [${step_name}] ${bench}: existing valid result at ${details_json}, skipping."
        return 0
    fi

    echo "[$(date)] [${step_name}] Evaluating ${bench} on ${merged_dir} as ${tag}"
    ${PYTHON} ${SHARED_SCRIPTS}/evaluate_model.py \
        --model_path "${merged_dir}" \
        --model_name "${tag}" \
        --output_dir "${RESULTS_DIR}" \
        --tensor_parallel_size 1 \
        --dp_size ${N_GPUS} \
        --benchmarks "${bench}" \
        2>&1 | tee "${LOG_DIR}/eval_${step_name}_${bench}.log"
}

for CKPT_DIR in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT_DIR}")
    MERGED_DIR="${HF_CKPT_DIR}/${STEP_NAME}"
    TAG="${RUN_NAME}_${STEP_NAME}"

    if [ ! -d "${MERGED_DIR}" ] || [ ! -f "${MERGED_DIR}/model.safetensors" ]; then
        echo "[$(date)] [${STEP_NAME}] Merged dir not ready, skip eval."
        continue
    fi

    run_eval_one "${MERGED_DIR}" "${TAG}" "math500" "${STEP_NAME}"
    run_eval_one "${MERGED_DIR}" "${TAG}" "gsm8k"   "${STEP_NAME}"
done

echo "[$(date)] ===== All Done ====="
echo "All artifacts under: ${RUN_DIR}"
echo "Eval results under:  ${RESULTS_DIR}"
