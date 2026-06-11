#!/bin/bash
set -euo pipefail

# ============================================================
# SFT Baseline for Cross-Tokenizer OPD
# Uses teacher-generated responses to train the student model.
#
# Pipeline:
#   1. Generate teacher responses and produce SFT training parquet
#   2. Train student (Phi-4-mini-instruct) via verl's FSDPSFTTrainer
#   3. Evaluate
# ============================================================

export PYTHONPATH="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"

# Paths
EASYOPD_ROOT="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD"
EXPERIMENT_DIR="${EASYOPD_ROOT}/experiments"
EXP_DIR="${EXPERIMENT_DIR}/01_cross_tokenizer_opd"
SHARED_SCRIPTS="${EXPERIMENT_DIR}/_shared/scripts"

# Eval results live under this method dir (so different methods are isolated).
METHOD_DIR="${EXP_DIR}/methods/sft"
RESULTS_DIR="${METHOD_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TEACHER_MODEL="/root/workspace/models/Qwen2.5-7B-Instruct"
STUDENT_MODEL="/root/workspace/models/Phi-4-mini-instruct"
RAW_DATASET="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/dataset/mixed_math_code_10k"

# Method-specific training data (teacher-generated SFT parquet) lives under
# the experiment dir, NOT the method dir, because in principle other methods
# in the same experiment could reuse the same teacher responses if desired.
TRAIN_DATA_DIR="${EXP_DIR}/train_data"
SFT_TRAIN_PARQUET="${TRAIN_DATA_DIR}/teacher_sft_train.parquet"

# ----------------------------------------------------------------
# All training outputs go to LOCAL DISK (/root) for fast IO.
#
# Directory layout:
#   /root/workspace/models/runs/
#   └── 01_cross_tokenizer_opd/      <- EXP_NAME
#       └── sft/                     <- METHOD
#           └── sft_phi4mini/        <- RUN_NAME
#               ├── fsdp/            <- FSDP sharded ckpts (training native)
#               │   ├── global_step_78/
#               │   └── global_step_156/
#               ├── hf/              <- merged HuggingFace ckpts (eval/inference)
#               │   ├── global_step_78/
#               │   └── global_step_156/
#               └── logs/            <- training/eval logs
# ----------------------------------------------------------------
EXP_NAME="01_cross_tokenizer_opd"
METHOD="sft"
RUN_NAME="sft_phi4mini"

RUNS_ROOT="/root/workspace/models/runs"
RUN_DIR="${RUNS_ROOT}/${EXP_NAME}/${METHOD}/${RUN_NAME}"
FSDP_CKPT_DIR="${RUN_DIR}/fsdp"   # FSDPSFTTrainer writes global_step_X/ here
HF_CKPT_DIR="${RUN_DIR}/hf"       # merged HF ckpts go here
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${FSDP_CKPT_DIR}" "${HF_CKPT_DIR}" "${LOG_DIR}"

# Backward-compat alias (some downstream scripts still read OUTPUT_DIR)
OUTPUT_DIR="${FSDP_CKPT_DIR}"

# Training hyperparameters
NUM_EPOCHS=2
LR=2e-5
BATCH_SIZE=128
MICRO_BATCH_SIZE=4
MAX_LENGTH=4160
N_GPUS=8
TEACHER_TP=1
TEACHER_DP=8  # Data parallelism for teacher generation (each DP worker uses TEACHER_TP GPUs)
TEACHER_GPU_OFFSET=0  # GPU index offset for teacher generation

# ============================================================
# Step 1: Generate teacher responses and produce SFT training parquet
# ============================================================
if [ ! -f "${SFT_TRAIN_PARQUET}" ]; then
    echo "[$(date)] ===== Step 1: Generating teacher responses & building SFT parquet (TP=${TEACHER_TP}, DP=${TEACHER_DP}) ====="
    mkdir -p "${TRAIN_DATA_DIR}"
    ${PYTHON} ${SHARED_SCRIPTS}/gen_teacher_resp_dp.py \
        --teacher_model "${TEACHER_MODEL}" \
        --raw_dataset "${RAW_DATASET}" \
        --output_parquet "${SFT_TRAIN_PARQUET}" \
        --teacher_tp ${TEACHER_TP} \
        --teacher_dp ${TEACHER_DP} \
        --gpu_offset ${TEACHER_GPU_OFFSET} \
        --max_model_len ${MAX_LENGTH} \
        --temperature 0.6 \
        --max_tokens ${MAX_LENGTH} \
        --top_p 0.95
    echo "[$(date)] ===== Step 1 Done ====="
fi

# ============================================================
# Step 2: Run SFT Training via verl's FSDPSFTTrainer
# ============================================================
echo "[$(date)] ===== Step 2: Starting SFT Training (verl FSDPSFTTrainer) ===="
echo "Teacher Model: ${TEACHER_MODEL} (used for data generation)"
echo "Student Model: ${STUDENT_MODEL}"
echo "Training Data: ${SFT_TRAIN_PARQUET}"
echo "FSDP ckpt out: ${FSDP_CKPT_DIR}"
echo "HF  ckpt out: ${HF_CKPT_DIR}"
echo "Log dir:      ${LOG_DIR}"

# ---- Skip training if already finished ----
# Final step under save_freq=${SAVE_FREQ?} cadence is the largest multiple of
# SAVE_FREQ that is <= total_steps. We don't know total_steps a priori, so we
# infer "finished" from: there exists any global_step_* AND the largest one is
# at least NUM_EPOCHS * SAVE_FREQ (a conservative heuristic; for current
# config NUM_EPOCHS=2, save_freq=78 -> expect global_step_156).
# To avoid over-engineering, we use a simple rule: if there is a 'latest'
# bookkeeping file written by verl after the last save, skip. Otherwise we
# fall back to "largest existing step >= NUM_EPOCHS * save_freq".
SAVE_FREQ=78
EXPECTED_FINAL_STEP=$(( NUM_EPOCHS * SAVE_FREQ ))
LARGEST_EXISTING_STEP=$(ls -d ${FSDP_CKPT_DIR}/global_step_* 2>/dev/null \
    | awk -F'global_step_' '{print $2}' | sort -n | tail -1)
LARGEST_EXISTING_STEP=${LARGEST_EXISTING_STEP:-0}

if [ "${FORCE_RETRAIN:-0}" = "1" ]; then
    echo "[$(date)] FORCE_RETRAIN=1, will retrain even if checkpoints exist."
    SKIP_TRAIN=0
elif [ "${LARGEST_EXISTING_STEP}" -ge "${EXPECTED_FINAL_STEP}" ]; then
    echo "[$(date)] Found existing checkpoint global_step_${LARGEST_EXISTING_STEP} (>= expected final step ${EXPECTED_FINAL_STEP}); skipping Step 2."
    echo "[$(date)] To force retraining, run with: FORCE_RETRAIN=1 bash $0"
    SKIP_TRAIN=1
else
    echo "[$(date)] Largest existing step = ${LARGEST_EXISTING_STEP} < expected final ${EXPECTED_FINAL_STEP}; will (re)run training."
    SKIP_TRAIN=0
fi

if [ "${SKIP_TRAIN}" = "0" ]; then
TRAIN_GPUS=${N_GPUS}

/opt/conda/envs/OpenAgentRL-sj/bin/torchrun --nproc_per_node=${TRAIN_GPUS} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${SFT_TRAIN_PARQUET}" \
    data.val_files="${SFT_TRAIN_PARQUET}" \
    data.train_batch_size=${BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    model.partial_pretrain="${STUDENT_MODEL}" \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=false \
    model.strategy=fsdp2 \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=${LR} \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    trainer.total_epochs=${NUM_EPOCHS} \
    trainer.project_name=easyopd-sft \
    trainer.experiment_name=${RUN_NAME} \
    trainer.default_local_dir="${FSDP_CKPT_DIR}" \
    trainer.logger="['console']" \
    trainer.seed=42 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=-1 \
    trainer.n_gpus_per_node=${TRAIN_GPUS} \
    trainer.nnodes=1 \
    hydra.run.dir="${LOG_DIR}/hydra" \
    hydra.output_subdir=null \
    hydra.job.chdir=false

echo "[$(date)] ===== Step 2: SFT Training Completed ===="
echo "FSDP ckpts saved to: ${FSDP_CKPT_DIR}"
fi  # end SKIP_TRAIN guard

# Clean up legacy Hydra outputs/ dir if any (created by previous launches
# before we redirected hydra.run.dir to ${LOG_DIR}).
LEGACY_HYDRA_DIR="${METHOD_DIR}/outputs"
if [ -d "${LEGACY_HYDRA_DIR}" ]; then
    echo "[$(date)] Removing legacy Hydra outputs dir: ${LEGACY_HYDRA_DIR}"
    rm -rf "${LEGACY_HYDRA_DIR}"
fi
# ============================================================
# Step 2.5: Merge ALL FSDP checkpoints to HuggingFace format
# Each global_step_X under ${FSDP_CKPT_DIR}/ is merged into
#   ${HF_CKPT_DIR}/global_step_X/
# so that different checkpoints can be distinguished and evaluated separately.
# ============================================================
# Collect all global_step_* checkpoints, sorted by step number (ascending).
# NOTE: do NOT use `sort -t_ -k3 -n` here. The full path contains many
# underscores already (01_cross_tokenizer_opd, sft_phi4mini, ...), so -k3 picks
# up the wrong field and the sort silently degenerates to lexicographic order
# (global_step_156 < global_step_78). Instead, extract the step number after
# the LAST `global_step_` and sort numerically.
shopt -s nullglob
_ALL_CKPTS_RAW=( ${FSDP_CKPT_DIR}/global_step_* )
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

echo "[$(date)] ===== Step 2.5: Merging ${#ALL_CKPTS[@]} FSDP checkpoint(s) to HuggingFace format ===="
for CKPT_DIR in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT_DIR}")  # e.g. global_step_78
    TARGET_DIR="${HF_CKPT_DIR}/${STEP_NAME}"

    if [ -f "${TARGET_DIR}/model.safetensors" ] || [ -f "${TARGET_DIR}/pytorch_model.bin" ]; then
        echo "[$(date)] [${STEP_NAME}] Already merged at ${TARGET_DIR}, skipping."
        continue
    fi

    echo "[$(date)] [${STEP_NAME}] Merging ${CKPT_DIR} -> ${TARGET_DIR}"
    ${PYTHON} ${SHARED_SCRIPTS}/merge_fsdp_checkpoint.py \
        --checkpoint_dir "${CKPT_DIR}" \
        --output_dir "${TARGET_DIR}" \
        2>&1 | tee "${LOG_DIR}/merge_${STEP_NAME}.log"
done
echo "[$(date)] ===== Step 2.5: Merge Completed ===="
# ============================================================
# Step 3: Evaluate every merged checkpoint
# Loads HF ckpts directly from local disk (${HF_CKPT_DIR}).
# Results are tagged with step name (e.g. SFT_Phi4mini_global_step_78)
# so different checkpoints can be distinguished.
# ============================================================
echo "[$(date)] ===== Step 3: Evaluating SFT Model(s) ===="

# Helper: returns 0 (true) if ${RESULTS_DIR}/<TAG>_<BENCH>_details.json exists
# AND its 'total' field is > 0 (i.e. the previous eval was not a failed run
# like the broken global_step_234 case that wrote total=0). Otherwise returns 1.
eval_already_done() {
    local details_json="$1"
    [ -f "${details_json}" ] || return 1
    local total
    total=$(${PYTHON} -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('total',0))" "${details_json}" 2>/dev/null || echo 0)
    [ "${total}" -gt 0 ]
}

run_eval_one() {
    # $1 = MERGED_MODEL_DIR, $2 = MODEL_TAG, $3 = BENCH (math500|gsm8k|...), $4 = STEP_NAME
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
    MERGED_MODEL_DIR="${HF_CKPT_DIR}/${STEP_NAME}"
    MODEL_TAG="${RUN_NAME}_${STEP_NAME}"

    if [ ! -d "${MERGED_MODEL_DIR}" ]; then
        echo "[$(date)] [${STEP_NAME}] Merged dir not found, skip evaluation."
        continue
    fi

    run_eval_one "${MERGED_MODEL_DIR}" "${MODEL_TAG}" "math500" "${STEP_NAME}"
    run_eval_one "${MERGED_MODEL_DIR}" "${MODEL_TAG}" "gsm8k"   "${STEP_NAME}"
done

echo "[$(date)] ===== All Done ====="
echo "All artifacts under: ${RUN_DIR}"
echo "Eval results under:  ${RESULTS_DIR}"
echo "(Tip: re-evaluate everything with: FORCE_REEVAL=1 bash $0)"