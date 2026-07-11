#!/bin/bash
set -euo pipefail
trap 'rc=$?; echo "[FATAL] launch.sh exited with code $rc at line $LINENO (last cmd: $BASH_COMMAND)" >&2' ERR

# ---- Mirror ALL stdout+stderr to a shared-disk log (readable from the dev machine) ----
# latest_launch.log always holds the most recent run; a timestamped copy is kept too.
_SDPO_LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
mkdir -p "${_SDPO_LOG_DIR}"
# Per-run log name so parallel instances on a shared disk don't overwrite each
# other's launch log. Pass a unique RUN_NAME for every sweep instance.
_SDPO_RUN_TAG="${RUN_NAME:-${DATASET:-run}}"
LAUNCH_LOG="${_SDPO_LOG_DIR}/${_SDPO_RUN_TAG}_latest_launch.log"
_SDPO_LOG_TS="${_SDPO_LOG_DIR}/${_SDPO_RUN_TAG}_launch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LAUNCH_LOG}" "${_SDPO_LOG_TS}") 2>&1
echo "[launch] Full log mirrored to (shared disk): ${LAUNCH_LOG}"

# ============================================================
# SDPO — Self-Distillation Policy Optimization
#   Paper: "Reinforcement Learning via Self-Distillation"
#          Hübotter et al., 2026 (arXiv:2601.20802)
#          Code:  https://github.com/lasgroup/SDPO
#
# SDPO augments GRPO with self-distillation: for each prompt the policy
# samples a *group* of rollouts (rollout.n>1); failed rollouts are reprompted
# with a correct demonstration from a successful rollout in the SAME group, and
# an EMA copy of the policy (the "self-teacher", initialised from the base model
# and EMA-updated at teacher_update_rate each step) re-scores the original
# response under that feedback-informed context. The student is distilled
# towards the stop-gradient self-teacher (paper Eq. 1) via logit-level top-K
# generalized JSD. Samples without a teacher get zero gradient (no GRPO
# fallback); token-level rollout-correction IS weights are applied. This
# mirrors the lasgroup/SDPO reference exactly. No external teacher model.
#
# Pipeline (mirrors 06_general_opd/grpo):
#   1. Prepare RL prompt parquet (train/val)
#   2. (Re)start Ray
#   3. SDPO training via verl.trainer.main_ppo (loss_mode=sdpo)
#   4. Merge each global_step_X/actor/ -> HF format
#   5. Evaluate every merged ckpt on math500 + gsm8k
# ============================================================

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
unset RAY_ADDRESS

# ----------------- Conda / Ray binaries -----------------
# Resolve python/ray of the CURRENTLY ACTIVE conda env.
# Prefer $CONDA_PREFIX (robust against `python` aliases / PATH ordering that can
# wrongly resolve to the base env inside a non-interactive `bash` subshell),
# then fall back to PATH lookup. Override anytime with PYTHON=/path/to/python.
if [ -z "${PYTHON:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYTHON="${CONDA_PREFIX}/bin/python"
    else
        PYTHON="$(command -v python || true)"
    fi
fi
if [ -z "${PYTHON}" ] || [ ! -x "${PYTHON}" ]; then
    echo "[FATAL] No python found. Pass your env explicitly, e.g.:" >&2
    echo "        PYTHON=/root/miniconda3/envs/srpo/bin/python bash <this script> ..." >&2
    exit 1
fi
# Ray defaults to the SAME env as PYTHON (override with RAY=/path/to/ray).
if [ -z "${RAY:-}" ]; then
    if [ -x "$(dirname "${PYTHON}")/ray" ]; then
        RAY="$(dirname "${PYTHON}")/ray"
    else
        RAY="$(command -v ray || true)"
    fi
fi
echo "[$(date)] Using PYTHON=${PYTHON}"
echo "[$(date)] Using RAY=${RAY}"
export PATH="$(dirname "${PYTHON}"):${PATH}"

# ----------------- Dependency preflight -----------------
# verl training needs a full stack (pandas/pyarrow/ray/vllm/verl). Fail fast and
# list everything that is missing, instead of crashing one package at a time.
echo "[$(date)] Checking training deps in: ${PYTHON} (rollout=${ROLLOUT_ENGINE:-vllm})"
_MISSING=$(ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}" ${PYTHON} - <<'PYEOF'
import importlib.util, os
# verl is provided in-repo via PYTHONPATH (set later), so it is not checked here.
# Only the selected rollout backend is required (hf needs neither vllm nor sglang).
req = ["pandas", "pyarrow", "datasets", "transformers", "torch", "ray"]
engine = os.environ.get("ROLLOUT_ENGINE", "vllm")
if engine in ("vllm", "sglang"):
    req.append(engine)
miss = [m for m in req if importlib.util.find_spec(m) is None]
print(" ".join(miss))
PYEOF
)
if [ -n "${_MISSING}" ]; then
    echo "[FATAL] '${PYTHON}' is missing required packages: ${_MISSING}" >&2
    echo "        This conda env is NOT a complete verl training environment." >&2
    echo "        Fix by either:" >&2
    echo "          (a) PYTHON=/path/to/verl-env/bin/python bash <this script> ... , or" >&2
    echo "          (b) install the verl training stack into the active env." >&2
    exit 1
fi

# ----------------- Paths -----------------
EASYOPD_ROOT="${EASYOPD_ROOT:-/path/to/workspace/EasyOPD}"
export PYTHONPATH="${EASYOPD_ROOT}:${PYTHONPATH:-}"
EXPERIMENT_DIR="${EASYOPD_ROOT}/experiments"
EXP_DIR="${EXPERIMENT_DIR}/04_self_opd"
SHARED_SCRIPTS="${EXPERIMENT_DIR}/_shared/scripts"

METHOD_DIR="${EXP_DIR}/methods/sdpo"
RESULTS_DIR="${METHOD_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Student = self-teacher (single model; no external teacher for SDPO).
STUDENT_MODEL="${STUDENT_MODEL:-/root/models/Qwen3-8B}"

# Raw dataset (shared) used to build the RL prompt parquet.
RAW_DATASET_DIR="${RAW_DATASET_DIR:-/path/to/workspace/workspace/dataset/mixed_math_code_10k}"

TRAIN_DATA_DIR="${EXP_DIR}/train_data"

# Dataset selector (each <name> expects <name>_{train,val}.parquet in train_data/
# and <name>_eval.parquet in _shared/eval_data/). The parquet's data_source drives
# BOTH the reward routing and the wandb metric val-(core|aux)/<data_source>/score/mean@VAL_N:
#   DATASET=chemistry|biology|material|physics -> SciKnowEval MCQ (data_source=sciknoweval;
#                     these share one metric val-aux/sciknoweval/score/mean@16).
#   DATASET=tooluse   -> agentic Action/Action-Input task (data_source=tooluse).
#   DATASET=gsm8k     -> grade-school MATH, \boxed{} answers (data_source=gsm8k).
#   DATASET=default   -> math/code RL prompts built from RAW_DATASET_DIR (Step 1).
DATASET="${DATASET:-default}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/path/to/EasyOPD/experiments/_shared/eval_data}"
if [ "${DATASET}" != "default" ]; then
    SDPO_TRAIN_PARQUET="${SDPO_TRAIN_PARQUET:-${TRAIN_DATA_DIR}/${DATASET}_train.parquet}"
    SDPO_VAL_PARQUET="${SDPO_VAL_PARQUET:-${TRAIN_DATA_DIR}/${DATASET}_val.parquet}"
    EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-${DATASET}}"
    EVAL_DATA_DIR="${EXPERIMENT_DIR}/_shared/eval_data"
    # Per-domain sets are small (~hundreds–few k prompts). Save often enough that a
    # checkpoint exists to merge, and validate every N steps so val metrics appear
    # in wandb. VAL_N=16 -> val-(core|aux)/<data_source>/score/mean@16.
    SAVE_FREQ="${SAVE_FREQ:-20}"
    # High so a re-run RESUMES long-horizon training instead of skipping straight to
    # eval (a completed short run is still handled by verl's internal step check).
    EXPECTED_FINAL_STEP="${EXPECTED_FINAL_STEP:-100000}"
    TEST_FREQ="${TEST_FREQ:-5}"
    VAL_N="${VAL_N:-16}"
    # Fail fast instead of silently falling back to the math/code default — a missing
    # parquet would otherwise train on the WRONG data and log the wrong val metric
    # (val-aux/<math|code>/...@1 instead of val-aux/<data_source>/score/mean@16).
    if [ ! -f "${SDPO_TRAIN_PARQUET}" ] || [ ! -f "${SDPO_VAL_PARQUET}" ]; then
        echo "[FATAL] DATASET=${DATASET} but its prompt parquets are missing:" >&2
        echo "          train: ${SDPO_TRAIN_PARQUET}" >&2
        echo "          val:   ${SDPO_VAL_PARQUET}" >&2
        echo "        Prepare them under ${TRAIN_DATA_DIR}/ first (mirror chemistry/biology)." >&2
        exit 1
    fi
fi
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-math500 gsm8k}"

# Training parquet: an explicit SDPO_TRAIN_PARQUET/SDPO_VAL_PARQUET overrides the
# Step-1 build (Step 1 is skipped automatically when these files already exist).
RL_TRAIN_PARQUET="${SDPO_TRAIN_PARQUET:-${TRAIN_DATA_DIR}/rl_prompts_train.parquet}"
RL_VAL_PARQUET="${SDPO_VAL_PARQUET:-${TRAIN_DATA_DIR}/rl_prompts_val.parquet}"
# SDPO-faithful reward: dispatch to the ported lasgroup/SDPO feedback scorers
# (verl/utils/reward_score/feedback) so the success signal that drives the
# self-distillation demonstrations matches the reference exactly.
REWARD_FN="${SHARED_SCRIPTS}/feedback_reward_fn.py"

# ----------------- Run dir layout -----------------
EXP_NAME="04_self_opd"
METHOD="sdpo"
# Separate checkpoint dir per dataset, so a math/code run and a chemistry run
# don't auto-resume each other's checkpoints (dataloader/sampler state is
# dataset-specific and resuming across datasets raises StopIteration).
_MODEL_TAG="$(basename "${STUDENT_MODEL}")"
if [ "${DATASET}" = "default" ]; then
    RUN_NAME="${RUN_NAME:-sdpo_${_MODEL_TAG}}"
else
    RUN_NAME="${RUN_NAME:-sdpo_${_MODEL_TAG}_${DATASET}}"
fi

RUNS_ROOT="${RUNS_ROOT:-${METHOD_DIR}/checkpoints}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
FSDP_CKPT_DIR="${RUN_DIR}/fsdp"
HF_CKPT_DIR="${RUN_DIR}/hf"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${FSDP_CKPT_DIR}" "${HF_CKPT_DIR}" "${LOG_DIR}"

# ----------------- Single-node Ray settings -----------------
RAY_NODE_IP="${RAY_NODE_IP:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_HEAD_ADDRESS="${RAY_NODE_IP}:${RAY_PORT}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
RAY_READY_RETRIES="${RAY_READY_RETRIES:-12}"

# ----------------- Hyperparameters -----------------
N_GPUS="${N_GPUS:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"         # align lasgroup SDPO (max_response_length=8192)
# align lasgroup SDPO (max_model_len=18944 = template 512 + prompt 2048 + feedback 8192 + response 8192)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18944}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"  # only used if USE_DYNAMIC_BSZ=True (SDPO uses static)

ACTOR_LR="${ACTOR_LR:-1e-5}"                          # align lasgroup SDPO (run_sdpo_all.sh: lr=1e-5)
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"              # align SDPO (lr_warmup_steps=10)
# NOTE: lasgroup SDPO trains 30 epochs on small per-domain datasets. EasyOPD's
# default dataset (mixed_math_code ~9.9k prompts) is much larger, so 2 epochs is
# a sensible default here. Set TOTAL_EPOCHS=30 to match the reference epoch count
# (and bump SAVE_FREQ/EXPECTED_FINAL_STEP accordingly for your dataset).
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"            # align SDPO (train_batch_size=32)
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"     # align SDPO (== train_batch -> on-policy, 1 grad step/batch)
# Static micro-batching (align SDPO: ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False).
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
# Under static batching, verl's validate_config requires rollout/ref log_prob
# micro-batch size (it gates on actor.use_dynamic_bsz). align SDPO user.yaml
# (rollout/ref.log_prob_micro_batch_size_per_gpu=1).
LOG_PROB_MICRO_BSZ_PER_GPU="${LOG_PROB_MICRO_BSZ_PER_GPU:-1}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"           # align SDPO user.yaml (clip_ratio_high=0.28; inert in pure SDPO)
EXPECTED_FINAL_STEP="${EXPECTED_FINAL_STEP:-308}"
SAVE_FREQ="${SAVE_FREQ:-77}"
TEST_FREQ="${TEST_FREQ:--1}"
# Run a pre-training (step-0) validation to log the baseline BEFORE any update.
# (lasgroup SDPO uses val_before_train=False; pass VAL_BEFORE_TRAIN=False to match it.)
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
# Validation samples per prompt -> enables val-core/.../{mean,best,maj}@VAL_N.
VAL_N="${VAL_N:-1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"             # align SDPO val_kwargs (temperature=0.6)
VAL_TOP_P="${VAL_TOP_P:-0.95}"                        # align SDPO val_kwargs (top_p=0.95)

# SDPO-specific (paper / easyopd.methods.sdpo)
ROLLOUT_N="${ROLLOUT_N:-8}"                          # >1 so each group has successes to reprompt from
SDPO_ALPHA="${SDPO_ALPHA:-0.5}"                       # 0=fwd KL, 1=rev KL, 0.5=symmetric JSD (paper-recommended)
SDPO_IS_CLIP="${SDPO_IS_CLIP:-2.0}"
SDPO_SUCCESS_THRESHOLD="${SDPO_SUCCESS_THRESHOLD:-0.5}"   # align SDPO actor.yaml (success_reward_threshold=0.5)
SDPO_MAX_REPROMPT_LEN="${SDPO_MAX_REPROMPT_LEN:-10240}"   # align SDPO (max_reprompt_len=10240)
SDPO_TOPK="${SDPO_TOPK:-100}"                         # logit-level top-K distillation (paper default)
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"

ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"             # rollout backend: vllm | sglang | hf
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"

ENABLE_GRAD_CKPT="${ENABLE_GRAD_CKPT:-True}"
FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-True}"
FSDP_OPTIMIZER_OFFLOAD="${FSDP_OPTIMIZER_OFFLOAD:-True}"
# FSDP master-weight dtype for the policy AND the EMA self-teacher (role=ref).
# align lasgroup SDPO: verl's FSDPEngineConfig.model_dtype default is "fp32"
# (fp32 master weights + bf16 autocast compute), which SDPO does not override.
# Set MODEL_DTYPE=bfloat16 to trade faithfulness for memory if you hit OOM.
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
# align lasgroup SDPO (actor.yaml: use_torch_compile=true).
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-True}"

# Logging backend. Default: console + Weights & Biases (run `wandb login` first,
# or export WANDB_API_KEY=...). Set WANDB=0 to disable wandb (console only).
WANDB="${WANDB:-1}"
if [ "${WANDB}" = "1" ]; then
    LOGGER_OVERRIDE='trainer.logger=["console","wandb"]'
else
    LOGGER_OVERRIDE='trainer.logger=["console"]'
fi

echo "[$(date)] SDPO: model=${STUDENT_MODEL} rollout_n=${ROLLOUT_N} alpha=${SDPO_ALPHA}"

# ============================================================
# Step 1: Prepare RL prompt parquet (train + val)
# ============================================================
if [ ! -f "${RL_TRAIN_PARQUET}" ] || [ ! -f "${RL_VAL_PARQUET}" ]; then
    echo "[$(date)] ===== Step 1: Building RL prompt parquet ====="
    mkdir -p "${TRAIN_DATA_DIR}"
    ${PYTHON} - <<PYEOF
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

DATASET_DIR = "${RAW_DATASET_DIR}"
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
        ["solve", "find", "calculate", "prove", "\\\\boxed"]) else "code"
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
    echo "[$(date)] [Step 1] RL prompt parquet exists, skipping."
fi

# ============================================================
# Step 2: (Re)start Ray on this node
# ============================================================
if [ "${SKIP_RAY_RESTART:-0}" != "1" ]; then
    echo "[$(date)] ===== Step 2: Restarting Ray ====="
    unset RAY_ADDRESS
    ${RAY} stop --force 2>/dev/null || true
    sleep 5
    rm -rf /tmp/ray/session_* /tmp/ray/ray_current_cluster /tmp/ray/*.json 2>/dev/null || true
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
import os, ray
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
[ "${ray_ready}" = "1" ] || { echo "[$(date)] ERROR: Ray health check failed."; exit 1; }

# ============================================================
# Step 3: SDPO Training
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
    SKIP_TRAIN=0
elif [ "${LARGEST_EXISTING_STEP}" -ge "${EXPECTED_FINAL_STEP}" ]; then
    echo "[$(date)] Found global_step_${LARGEST_EXISTING_STEP} >= ${EXPECTED_FINAL_STEP}; skipping training."
    SKIP_TRAIN=1
else
    SKIP_TRAIN=0
fi

if [ "${SKIP_TRAIN}" = "0" ]; then
echo "[$(date)] ===== Step 3: SDPO Training ====="
TRAIN_LAUNCH_CWD="/tmp/easyopd_${EXP_NAME}_${METHOD}_${RUN_NAME}"
mkdir -p "${TRAIN_LAUNCH_CWD}"

(
cd "${TRAIN_LAUNCH_CWD}"
${PYTHON} -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.rollout_correction.rollout_is=token \
    +algorithm.rollout_correction.rollout_is_threshold=2.0 \
    "data.train_files=['${RL_TRAIN_PARQUET}']" \
    "data.val_files=['${RL_VAL_PARQUET}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRAD_CKPT} \
    actor_rollout_ref.actor.use_torch_compile=${USE_TORCH_COMPILE} \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    +actor_rollout_ref.actor.self_distillation.alpha=${SDPO_ALPHA} \
    +actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
    +actor_rollout_ref.actor.self_distillation.distillation_topk=${SDPO_TOPK} \
    +actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
    +actor_rollout_ref.actor.self_distillation.is_clip=${SDPO_IS_CLIP} \
    +actor_rollout_ref.actor.self_distillation.success_reward_threshold=${SDPO_SUCCESS_THRESHOLD} \
    +actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    +actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
    +actor_rollout_ref.actor.self_distillation.max_reprompt_len=${SDPO_MAX_REPROMPT_LEN} \
    +actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    +actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    +actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=True \
    +actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ_PER_GPU} \
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    "${LOGGER_OVERRIDE}" \
    trainer.project_name=easyopd-sdpo \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default_local_dir="${FSDP_CKPT_DIR}" \
    "${@:1}" \
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
[ ${#ALL_CKPTS[@]} -gt 0 ] || { echo "[$(date)] ERROR: No checkpoint in ${FSDP_CKPT_DIR}"; exit 1; }

echo "[$(date)] ===== Step 4: Merging ${#ALL_CKPTS[@]} checkpoint(s) ====="
for CKPT_DIR in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT_DIR}")
    ACTOR_DIR="${CKPT_DIR}/actor"
    TARGET_DIR="${HF_CKPT_DIR}/${STEP_NAME}"
    [ -d "${ACTOR_DIR}" ] || { echo "[${STEP_NAME}] no actor/, skip"; continue; }
    if [ -f "${TARGET_DIR}/model.safetensors" ] || [ -f "${TARGET_DIR}/pytorch_model.bin" ]; then
        echo "[${STEP_NAME}] already merged, skip"; continue
    fi
    echo "[$(date)] [${STEP_NAME}] Merging -> ${TARGET_DIR}"
    ${PYTHON} ${SHARED_SCRIPTS}/merge_fsdp.py \
        --ckpt_dir "${ACTOR_DIR}" \
        --base_model "${STUDENT_MODEL}" \
        --output_dir "${TARGET_DIR}" \
        2>&1 | tee "${LOG_DIR}/merge_${STEP_NAME}.log"
done
echo "[$(date)] ===== Step 4 Done ====="

# ============================================================
# Step 5: Evaluate every merged checkpoint
# ============================================================
echo "[$(date)] ===== Step 5: Evaluating SDPO checkpoint(s) ====="
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
        echo "[${step_name}] ${bench}: cached, skip"; return 0
    fi
    echo "[$(date)] [${step_name}] Eval ${bench} on ${merged_dir}"
    ${PYTHON} ${SHARED_SCRIPTS}/evaluate_model.py \
        --model_path "${merged_dir}" \
        --model_name "${tag}" \
        --output_dir "${RESULTS_DIR}" \
        --tensor_parallel_size 1 \
        --dp_size ${N_GPUS} \
        --benchmarks "${bench}" \
        --data_dir "${EVAL_DATA_DIR}" \
        2>&1 | tee "${LOG_DIR}/eval_${step_name}_${bench}.log"
}
for CKPT_DIR in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT_DIR}")
    MERGED_DIR="${HF_CKPT_DIR}/${STEP_NAME}"
    TAG="${RUN_NAME}_${STEP_NAME}"
    [ -f "${MERGED_DIR}/model.safetensors" ] || { echo "[${STEP_NAME}] not merged, skip eval"; continue; }
    for _bench in ${EVAL_BENCHMARKS}; do
        run_eval_one "${MERGED_DIR}" "${TAG}" "${_bench}" "${STEP_NAME}"
    done
done

echo "[$(date)] ===== All Done ====="
echo "Artifacts: ${RUN_DIR}"
echo "Results:   ${RESULTS_DIR}"
