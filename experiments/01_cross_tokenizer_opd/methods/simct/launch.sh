#!/bin/bash
set -euo pipefail
trap 'rc=$?; echo "[FATAL] launch.sh exited with code $rc at line $LINENO (last cmd: $BASH_COMMAND)" >&2' ERR

# ============================================================
# SimCT (Cross-Tokenizer OPD) — span-based KD, KDFlow-aligned config
#
# Mirrors the structure of methods/simple/launch.sh, swapping the loss mode
# from `simple` (overlap-vocab reverse KL on aligned single tokens) to
# `simct` (span-level virtual-vocabulary KD, KDFlow's `span_ctkd`).
#
# The teacher sidecar / hidden-state transport is shared with `simple`:
# `easyopd.methods.simct.losses.register_simct_loss` registers `simct` and
# `span_ctkd` as cross-tokenizer distillation losses that consume the same
# `+distillation.simple_teacher_*` fields. Do not rename those fields.
#
# Teacher backend: ALWAYS SGLang (the +distillation.teacher_models.*.inference.name
# field is purely informational; teacher_sidecar.py constructs SGLangEngineService
# unconditionally and never reads it). Do NOT try to switch teacher to vLLM.
#
# Pipeline:
#   1. Prepare RL prompt parquet (train.parquet / val.parquet) — shared with
#      methods/simple/, so this step is normally a no-op cache hit.
#   2. (Re)start Ray
#   3. GRPO + cross-tokenizer KD (simct loss) via verl.trainer.main_ppo
#      with distillation.* enabled. Teacher and student share GPUs by default
#      (8S+8T colocated, sleep/wake gated). Set TEACHER_LAYOUT=split to fall
#      back to the legacy 6 student + 2 teacher physical-split layout.
#   4. Merge each global_step_X/actor/ -> HF format
#   5. Evaluate every merged ckpt on math500 + gsm8k
#
# Hyperparameters dispatched by TEACHER_LAYOUT (default = shared):
#   shared (8S+8T colocated, KDFlow-aligned):
#     train_batch_size / ppo_mini_batch_size = 64 (divisible by 8 student ranks)
#     EXPECTED_FINAL_STEP = (9900 / 64) * 2 ≈ 308
#     ROLLOUT_GPU_MEM_UTIL  = 0.55  (vLLM gpu_memory_utilization)
#     TEACHER_GPU_MEM_UTIL  = 0.55  (SGLang mem_fraction_static)
#     SIMPLE_TEACHER_NUM_GPUS_PER_ACTOR = 0 (verl-friendly; binding via base_gpu_id)
#     vLLM and SGLang are mutually-exclusive at any given step phase via
#     verl FSDPVLLMShardingManager (vLLM auto-sleep) + EasyOPD sidecar wake/sleep,
#     so 0.55 + 0.55 > 1.0 is safe (only one is awake at a time).
#   split (legacy 6 student + 2 teacher physical split):
#     train_batch_size / ppo_mini_batch_size = 66 (divisible by 6 student ranks)
#     EXPECTED_FINAL_STEP = (9900 / 66) * 2 = 300
#     ROLLOUT_GPU_MEM_UTIL  = 0.25, TEACHER_GPU_MEM_UTIL = 0.6
#     SIMPLE_TEACHER_NUM_GPUS_PER_ACTOR = 1.0 (full Ray GPU per teacher actor)
#
# Other knobs (shared by both layouts):
#   max_prompt / max_response   = 4096 / 4096   (max_model_len = 8193)
#   ppo_max_token_len_per_gpu   = 16384         (default; lower this if FSDP unshard OOMs)
#   actor_lr / warmup_ratio     = 5e-7 / 0.05   (aligned with KDFlow)
#   rollout.n / temperature     = 1 / 0.6       (OPD doesn't need multi-sample)
#   total_epochs                = 1
#   save_freq                   = 20            (-> ckpts at step 20, 40, ..., 140, 154)
#   FSDP param/optimizer offload= True (always on, mandatory for shared mode
#                                       to leave room for SGLang teacher KV cache).
#
# OOM mitigation (shared mode): if an OOM still occurs after sidecar wake,
# step down TEACHER_GPU_MEM_UTIL: 0.55 → 0.45 → 0.35. If FSDP unshard peak
# itself OOMs, keep param_offload/optimizer_offload True (already default).
#
# SimCT vs Simple — what is different and what is identical:
#   * IDENTICAL: teacher backend (SGLang), GPU layout, RL prompt parquet,
#                checkpoint dir layout, FSDP/Ray boilerplate, eval pipeline.
#   * DIFFERENT:
#       - DISTILL_LOSS_MODE=simct       (vs simple)
#       - actor_rollout_ref.actor.policy_loss.loss_mode=simct
#       - distillation.distillation_loss.loss_mode=simct
#       - dropped `+actor_rollout_ref.actor.policy_loss.simple_kl_direction`
#         (simct ignores it; KL direction is taken from
#         distillation.distillation_loss.cross_tokenizer_kl_direction).
#       - `simple_loss_clamp` is RETAINED because verl/dp_actor.py applies
#         it to both `simple` and `simct` modes (see dp_actor.py:1101).
# ============================================================

export PYTHONPATH="/path/to/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

# Do not export RAY_ADDRESS before starting Ray. If raylet inherits
# RAY_ADDRESS=auto, its prestarted Python workers may fail to register and
# leave verl.trainer.main_ppo stuck inside ray.init().
unset RAY_ADDRESS

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
RAY="/opt/conda/envs/OpenAgentRL-sj/bin/ray"
export PATH="/opt/conda/envs/OpenAgentRL-sj/bin:${PATH}"

# Single-node Ray settings. Keep CPU count moderate: Ray prestarts one Python
# worker per CPU, and starting 64 workers at once was observed to make workers
# miss their registration timeout on this machine.
RAY_NODE_IP="${RAY_NODE_IP:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_HEAD_ADDRESS="${RAY_NODE_IP}:${RAY_PORT}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
RAY_READY_RETRIES="${RAY_READY_RETRIES:-12}"

# ----------------- Paths -----------------
EASYOPD_ROOT="/path/to/EasyOPD"
EXPERIMENT_DIR="${EASYOPD_ROOT}/experiments"
EXP_DIR="${EXPERIMENT_DIR}/01_cross_tokenizer_opd"
SHARED_SCRIPTS="${EXPERIMENT_DIR}/_shared/scripts"

METHOD_DIR="${EXP_DIR}/methods/simct"
RESULTS_DIR="${METHOD_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Student = SFT-warmed phi4-mini-instruct (1 epoch, mid-epoch ckpt at step 78).
# After flattening (see sft/launch.sh fix), the SFT exporter writes two parallel
# HF dirs:
#   .../hf/global_step_78/   <- mid-epoch ckpt (this one, used as RL start)
#   .../hf/global_step_156/  <- final ckpt
STUDENT_MODEL="/path/to/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini/hf/global_step_116"
# Teacher = original Qwen2.5-7B-Instruct on local disk.
TEACHER_MODEL="/path/to/models/Qwen2.5-7B-Instruct"

# RL prompt data: experiment-level shared (simple/simct/uld/dskd... can reuse).
TRAIN_DATA_DIR="${EXP_DIR}/train_data"
RL_TRAIN_PARQUET="${TRAIN_DATA_DIR}/rl_prompts_train.parquet"
RL_VAL_PARQUET="${TRAIN_DATA_DIR}/rl_prompts_val.parquet"

REWARD_FN="${SHARED_SCRIPTS}/reward_fn.py"

# ----------------- Run dir layout (mirrors sft / simple) -----------------
EXP_NAME="01_cross_tokenizer_opd"
METHOD="simct"
RUN_NAME="simct_phi4mini"

RUNS_ROOT="/path/to/models/runs"
RUN_DIR="${RUNS_ROOT}/${EXP_NAME}/${METHOD}/${RUN_NAME}"
FSDP_CKPT_DIR="${RUN_DIR}/fsdp"
HF_CKPT_DIR="${RUN_DIR}/hf"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${FSDP_CKPT_DIR}" "${HF_CKPT_DIR}" "${LOG_DIR}"

# ----------------- Hyperparameters (KDFlow reference) -----------------
N_GPUS=8
# Layout selector: "shared" (8S+8T colocated, default) or "split" (6S+2T legacy).
TEACHER_LAYOUT="${TEACHER_LAYOUT:-shared}"

MAX_PROMPT_LEN=4096
MAX_RESPONSE_LEN=4096
MAX_MODEL_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1 ))   # 8193
# Per-GPU token budget for FSDP forward/backward and vLLM log-prob recompute.
# Higher = fewer micro-batches per step, faster update_actor, but higher peak
# activation memory. Default 16384 packs ~2 full sequences per micro-batch.
# Used to be 8193 (one seq/micro-batch, lowest mem) before SPEED_TIER tuning.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

ACTOR_LR=5e-7
LR_WARMUP_RATIO=0.05
TOTAL_EPOCHS=1

SAVE_FREQ=20                  # ckpts at step 20, 40, 60, ... (aligned with KDFlow save_steps=20)
TEST_FREQ=-1                  # eval after merge, not in-loop

ROLLOUT_TP=1
ROLLOUT_TEMPERATURE=0.6
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}
TEACHER_TP=1

# ----------------- Speed/Memory tier -----------------
# H20 has 97 GB/GPU but at the original "safe" tier (gradient_checkpointing=True,
# vLLM mem=0.55, SGLang mem=0.55, ppo_max_token=8193) only ~25 GB/GPU was used,
# leaving ~70 GB idle. SPEED_TIER lets you trade headroom for throughput.
#
# Concrete optimizations (referenced as A/B/C/D/E in design discussions):
#   A: gradient_checkpointing  True -> False    (~20% faster update_actor, +3x activation mem)
#   B: vLLM gpu_memory_util    0.55 -> 0.65     (~15% faster rollout, larger KV cache)
#   C: SGLang mem_fraction     0.55 -> 0.70     (~10% faster teacher prefill)
#   D: ppo_max_token_len/gpu   8193 -> 16384    (~10% faster update_actor, fewer micro-batches)
#   E: FSDP optimizer_offload  True -> False    (~5% faster, but +Adam state on GPU)
#
# Tier matrix:
#   safe       - none (original conservative settings, ~25 GB peak, ~66 s/step)
#   fast       - B + C + D                       (~50 GB peak, ~45 s/step, low OOM risk)
#   aggressive - A + B + C + D                   (~75 GB peak, ~30-35 s/step, recommended)
#                Default. Matches the user-validated ABCD combo from `simple`.
#                FSDP offload (E) intentionally kept ON for stability;
#                flip to "extreme" if needed.
#   extreme    - A + B + C + D + E               (~85 GB peak, ~25-30 s/step, OOM risk)
SPEED_TIER="${SPEED_TIER:-aggressive}"

if [ "${SPEED_TIER}" = "safe" ]; then
    ENABLE_GRAD_CKPT=True
    ROLLOUT_MEM_DEFAULT=0.35
    TEACHER_MEM_DEFAULT=0.35
    FSDP_PARAM_OFFLOAD=True
    FSDP_OPTIMIZER_OFFLOAD=True
elif [ "${SPEED_TIER}" = "fast" ]; then
    ENABLE_GRAD_CKPT=True
    ROLLOUT_MEM_DEFAULT=0.65
    TEACHER_MEM_DEFAULT=0.70
    FSDP_PARAM_OFFLOAD=True
    FSDP_OPTIMIZER_OFFLOAD=True
elif [ "${SPEED_TIER}" = "aggressive" ]; then
    # A + B + C + D: gradient checkpointing OFF, mem fractions raised,
    # ppo_max_token doubled. FSDP param/optimizer offload still ON so the
    # FSDP unshard peak stays bounded. This is the recommended tier for
    # H20-97GB when ~70 GB/GPU is otherwise idle.
    ENABLE_GRAD_CKPT=False
    ROLLOUT_MEM_DEFAULT=0.65
    TEACHER_MEM_DEFAULT=0.70
    FSDP_PARAM_OFFLOAD=True
    FSDP_OPTIMIZER_OFFLOAD=True
elif [ "${SPEED_TIER}" = "extreme" ]; then
    # A + B + C + D + E: also turns off optimizer offload. ~85 GB peak/GPU.
    # If update_actor OOMs, fall back to "aggressive".
    ENABLE_GRAD_CKPT=False
    ROLLOUT_MEM_DEFAULT=0.65
    TEACHER_MEM_DEFAULT=0.70
    FSDP_PARAM_OFFLOAD=True
    FSDP_OPTIMIZER_OFFLOAD=False
else
    echo "[FATAL] Unknown SPEED_TIER='${SPEED_TIER}', expected 'safe' | 'fast' | 'aggressive' | 'extreme'." >&2
    exit 1
fi
echo "[$(date)] SPEED_TIER=${SPEED_TIER}: grad_ckpt=${ENABLE_GRAD_CKPT}, rollout_mem=${ROLLOUT_MEM_DEFAULT}, teacher_mem=${TEACHER_MEM_DEFAULT}, ppo_max_token=${PPO_MAX_TOKEN_LEN_PER_GPU}, param_offload=${FSDP_PARAM_OFFLOAD}, optim_offload=${FSDP_OPTIMIZER_OFFLOAD}"

# Distillation knobs (KDFlow `span_ctkd` = SimCT span-level cross-tokenizer KD).
# In simct mode, `cross_tokenizer_kl_direction` is the authoritative source of
# the KL direction; the actor-side `simple_kl_direction` is ignored.
DISTILL_LOSS_MODE=simct
KL_DIRECTION=reverse
USE_POLICY_GRADIENT=False     # kd_ratio=1.0 -> pure distillation, no policy gradient term

# ----------------- Layout-specific dispatch -----------------
if [ "${TEACHER_LAYOUT}" = "shared" ]; then
    # 8 student + 8 teacher colocated on the same 8 GPUs (KDFlow recipe).
    # vLLM rollout (gpu_memory_utilization=0.55) and SGLang teacher
    # (mem_fraction_static=0.55) are mutually-exclusive in time:
    #   - verl FSDPVLLMShardingManager auto-sleeps vLLM after generate_sequences;
    #   - sidecar wakes SGLang only inside the simct/simple xtok batch builder.
    TEACHER_WORLD_SIZE=8
    TEACHER_GPU_IDS="0,1,2,3,4,5,6,7"
    TEACHER_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    SIMPLE_TEACHER_SHARE_STUDENT_POOL=True
    # CRITICAL: must be 0 here. verl creates a strict 8-GPU PlacementGroup
    # for the actor_rollout_ref worker (trainer.n_gpus_per_node=8) and its
    # _check_resource_available() reads ray.state.available_resources_per_node()
    # right after PG creation. If teacher actors already pre-occupy any
    # fractional GPU (e.g. 0.2 * 8 = 1.6), verl sees "available 6.4 < desired 8"
    # and aborts. By requesting 0 Ray GPU per teacher actor, the teacher
    # stays off Ray's GPU ledger entirely; physical binding is enforced via
    # `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + per-actor `base_gpu_id`
    # (already set in teacher_group.py). KDFlow uses 0.2 instead because
    # KDFlow controls the entire student GPU ledger itself; verl does not.
    SIMPLE_TEACHER_NUM_GPUS_PER_ACTOR=0
    TRAIN_BATCH_SIZE=64
    PPO_MINI_BATCH_SIZE=64
    EXPECTED_FINAL_STEP=154       # 9900 / 64 = 154 per epoch * 1 epoch = 154
    ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_MEM_DEFAULT}
    TEACHER_GPU_MEM_UTIL=${TEACHER_MEM_DEFAULT}
    STUDENT_N_GPUS_PER_NODE=8
elif [ "${TEACHER_LAYOUT}" = "split" ]; then
    # Legacy 6 student + 2 teacher physical split.
    # Reserve a FULL GPU per teacher actor in Ray's resource accounting so
    # Ray treats GPU 6/7 as fully occupied; student PG lands strictly on 0-5.
    TEACHER_WORLD_SIZE=2
    TEACHER_GPU_IDS="6,7"
    TEACHER_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    SIMPLE_TEACHER_SHARE_STUDENT_POOL=False
    SIMPLE_TEACHER_NUM_GPUS_PER_ACTOR=1.0
    TRAIN_BATCH_SIZE=66
    PPO_MINI_BATCH_SIZE=66
    EXPECTED_FINAL_STEP=150       # 9900 / 66 = 150 per epoch * 1 epoch = 150
    ROLLOUT_GPU_MEM_UTIL=0.25
    TEACHER_GPU_MEM_UTIL=0.6
    STUDENT_N_GPUS_PER_NODE=$((N_GPUS - TEACHER_WORLD_SIZE))   # 6
else
    echo "[FATAL] Unknown TEACHER_LAYOUT='${TEACHER_LAYOUT}', expected 'shared' or 'split'." >&2
    exit 1
fi

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
    sleep 3
    rm -rf /tmp/ray/session_* /tmp/ray/ray_current_cluster 2>/dev/null || true
    ${RAY} start --head --disable-usage-stats --node-ip-address=${RAY_NODE_IP} --port=${RAY_PORT} --num-cpus=${RAY_NUM_CPUS} --num-gpus=${N_GPUS} --include-dashboard=false
fi

export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
echo "[$(date)] Ray address: ${RAY_ADDRESS} (num_cpus=${RAY_NUM_CPUS}, num_gpus=${N_GPUS})"

ray_ready=0
for attempt in $(seq 1 ${RAY_READY_RETRIES}); do
    if ${PYTHON} - <<'PYEOF'
import os
import ray

ray_address = os.environ["RAY_ADDRESS"]
ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)

@ray.remote
def ping():
    return "ok"

assert ray.get(ping.remote(), timeout=10) == "ok"
print(f"[RayCheck] ray.init({ray_address}) + remote ping ok")
ray.shutdown()
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
# Step 3: GRPO + SimCT cross-tokenizer KD training
# Skip if largest existing global_step_X >= EXPECTED_FINAL_STEP.
# ============================================================
# Note: under `set -euo pipefail`, a failing `ls` (no matches) inside $(...)
# will silently kill the script. Use a bash for-loop over a glob instead.
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
    echo "[$(date)] To force retraining, run with: FORCE_RETRAIN=1 bash $0"
    SKIP_TRAIN=1
else
    echo "[$(date)] Largest existing step = ${LARGEST_EXISTING_STEP} < expected ${EXPECTED_FINAL_STEP}; will (re)run training."
    SKIP_TRAIN=0
fi

if [ "${SKIP_TRAIN}" = "0" ]; then
echo "[$(date)] ===== Step 3: GRPO + SimCT Cross-Tok KD Training ====="
echo "Student:        ${STUDENT_MODEL}"
echo "Teacher:        ${TEACHER_MODEL}"
echo "Train parquet:  ${RL_TRAIN_PARQUET}"
echo "Val parquet:    ${RL_VAL_PARQUET}"
echo "FSDP ckpt out:  ${FSDP_CKPT_DIR}"
if [ "${TEACHER_LAYOUT}" = "shared" ]; then
    echo "GPU split:      8 student + 8 teacher (shared, SGLang backend)"
else
    echo "GPU split:      ${STUDENT_N_GPUS_PER_NODE} student + ${TEACHER_WORLD_SIZE} teacher (split, SGLang backend)"
fi
echo "batch=${TRAIN_BATCH_SIZE}, mini=${PPO_MINI_BATCH_SIZE}, lr=${ACTOR_LR}, epochs=${TOTAL_EPOCHS}"
echo "max_prompt=${MAX_PROMPT_LEN}, max_response=${MAX_RESPONSE_LEN}, max_model_len=${MAX_MODEL_LEN}"

TRAIN_LAUNCH_CWD="/tmp/easyopd_${EXP_NAME}_${METHOD}_${RUN_NAME}"
mkdir -p "${TRAIN_LAUNCH_CWD}"
echo "[$(date)] Training driver cwd: ${TRAIN_LAUNCH_CWD}"

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
    actor_rollout_ref.actor.policy_loss.loss_mode=${DISTILL_LOSS_MODE} \
    +actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0 \
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
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name=compute_score \
    +distillation.enabled=True \
    +distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
    +distillation.nnodes=1 \
    "+distillation.simple_teacher_gpu_ids=[${TEACHER_GPU_IDS}]" \
    "+distillation.simple_teacher_visible_devices=[${TEACHER_VISIBLE_DEVICES}]" \
    +distillation.simple_teacher_share_student_pool=${SIMPLE_TEACHER_SHARE_STUDENT_POOL} \
    +distillation.simple_teacher_num_gpus_per_actor=${SIMPLE_TEACHER_NUM_GPUS_PER_ACTOR} \
    +distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    +distillation.teacher_models.teacher_model.num_replicas=${TEACHER_WORLD_SIZE} \
    +distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP} \
    +distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size=1 \
    +distillation.teacher_models.teacher_model.inference.name=sglang \
    +distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL} \
    +distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN} \
    +distillation.distillation_loss.loss_mode=${DISTILL_LOSS_MODE} \
    +distillation.distillation_loss.use_cross_tokenizer=True \
    +distillation.distillation_loss.use_task_rewards=False \
    +distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT} \
    +distillation.distillation_loss.distillation_loss_coef=1.0 \
    +distillation.distillation_loss.loss_max_clamp=10.0 \
    +distillation.distillation_loss.cross_tokenizer_kl_direction=${KL_DIRECTION} \
    trainer.balance_batch=True \
    'trainer.logger=["console"]' \
    trainer.project_name=easyopd-simct \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=${STUDENT_N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default_local_dir="${FSDP_CKPT_DIR}" \
    2>&1
) | tee "${LOG_DIR}/train.log"

echo "[$(date)] ===== Step 3: Training Completed ====="
fi  # SKIP_TRAIN guard

# ============================================================
# Step 4: Merge each global_step_X/actor/ -> ${HF_CKPT_DIR}/global_step_X/
# Stop Ray first to free GPU memory for the merge process.
# ============================================================
${RAY} stop --force 2>/dev/null || true

# Same caveat as Step 3: avoid `ls` failure under `set -euo pipefail`.
shopt -s nullglob
_ALL_CKPTS_RAW=( "${FSDP_CKPT_DIR}"/global_step_* )
shopt -u nullglob
# Sort numerically by step id.
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
    STEP_NAME=$(basename "${CKPT_DIR}")          # e.g. global_step_77
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
echo "[$(date)] ===== Step 5: Evaluating SimCT checkpoint(s) ====="

eval_already_done() {
    local details_json="$1"
    [ -f "${details_json}" ] || return 1
    local total
    total=$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('total',0))" "${details_json}" 2>/dev/null || echo 0)
    [ "${total}" -gt 0 ]
}

run_eval_one() {
    # $1 = MERGED_DIR, $2 = TAG, $3 = BENCH, $4 = STEP_NAME
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

    if [ ! -d "${MERGED_DIR}" ] || { [ ! -f "${MERGED_DIR}/model.safetensors" ] && [ ! -f "${MERGED_DIR}/model.safetensors.index.json" ]; }; then
        echo "[$(date)] [${STEP_NAME}] Merged dir not ready, skip eval."
        continue
    fi

    run_eval_one "${MERGED_DIR}" "${TAG}" "math500" "${STEP_NAME}"
    run_eval_one "${MERGED_DIR}" "${TAG}" "gsm8k"   "${STEP_NAME}"
done

echo "[$(date)] ===== All Done ====="
echo "All artifacts under: ${RUN_DIR}"
echo "Eval results under:  ${RESULTS_DIR}"
echo "(Tip: re-evaluate everything with: FORCE_REEVAL=1 bash $0)"
echo "(Tip: force retraining: FORCE_RETRAIN=1 bash $0)"
