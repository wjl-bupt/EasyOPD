#!/bin/bash
set -euo pipefail

# ============================================================
# SFT Training for Cross-Tokenizer OPD (KDFlow-aligned)
# ============================================================
# Trains Phi-4-mini-instruct on teacher-generated SFT data using
# verl's FSDPSFTTrainer, with all hyperparameters aligned to KDFlow.
#
# KDFlow reference config: scripts/sft/phi4_sft_warmup_10k_qwen.yaml
#   - learning_rate: 2e-6
#   - cutoff_len: 2048
#   - warmup_ratio: 0.05
#   - weight_decay: 0 (default)
#   - per_device_train_batch_size: 4
#   - gradient_accumulation_steps: 4
#   - 8 GPUs -> effective batch_size = 128
#   - num_train_epochs: 2.0
#   - lr_scheduler_type: cosine
#   - save_steps: 20
#   - packing: false
#   - bf16: true, pure_bf16: true
#   - gradient_checkpointing: true
#
# Prerequisites:
#   - Teacher SFT data generated via gen_teacher_sft_data.sh
#   - Output: teacher_sft_train.parquet in train_data/
# ============================================================

export PYTHONPATH="/path/to/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1

PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"

# Paths
EASYOPD_ROOT="/path/to/EasyOPD"
EXPERIMENT_DIR="${EASYOPD_ROOT}/experiments"
EXP_DIR="${EXPERIMENT_DIR}/01_cross_tokenizer_opd"
SHARED_SCRIPTS="${EXPERIMENT_DIR}/_shared/scripts"
METHOD_DIR="${EXP_DIR}/methods/sft"

STUDENT_MODEL="/path/to/models/Phi-4-mini-instruct"
TRAIN_DATA_DIR="${EXP_DIR}/train_data"
SFT_TRAIN_PARQUET="${TRAIN_DATA_DIR}/teacher_sft_train.parquet"

# ----------------------------------------------------------------
# Output directory layout (local disk for fast IO)
# ----------------------------------------------------------------
EXP_NAME="01_cross_tokenizer_opd"
METHOD="sft"
RUN_NAME="sft_phi4mini"

RUNS_ROOT="/path/to/models/runs"
RUN_DIR="${RUNS_ROOT}/${EXP_NAME}/${METHOD}/${RUN_NAME}"
FSDP_CKPT_DIR="${RUN_DIR}/fsdp"
HF_CKPT_DIR="${RUN_DIR}/hf"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${FSDP_CKPT_DIR}" "${HF_CKPT_DIR}" "${LOG_DIR}"

# ============================================================
# Training hyperparameters (aligned with KDFlow)
# ============================================================
NUM_EPOCHS=2
LR=2e-6                    # KDFlow: 2e-6 (NOT 2e-5!)
BATCH_SIZE=128             # KDFlow: 4 * 4 * 8 = 128
MICRO_BATCH_SIZE=4         # KDFlow: per_device_train_batch_size=4
MAX_LENGTH=2048            # KDFlow: cutoff_len=2048
WARMUP_RATIO=0.05          # KDFlow: warmup_ratio=0.05
WEIGHT_DECAY=0.0           # KDFlow: default (0)
LR_SCHEDULER=cosine        # KDFlow: cosine
SAVE_FREQ=20               # KDFlow: save_steps=20
N_GPUS=8
SEED=42

echo "[$(date)] ============================================================"
echo "[$(date)] SFT Training (KDFlow-aligned)"
echo "[$(date)] ============================================================"
echo "[$(date)] Student Model: ${STUDENT_MODEL}"
echo "[$(date)] Training Data: ${SFT_TRAIN_PARQUET}"
echo "[$(date)] FSDP ckpt dir: ${FSDP_CKPT_DIR}"
echo "[$(date)] HF ckpt dir:   ${HF_CKPT_DIR}"
echo "[$(date)] Log dir:       ${LOG_DIR}"
echo "[$(date)] Hyperparameters:"
echo "[$(date)]   LR=${LR}, epochs=${NUM_EPOCHS}, batch=${BATCH_SIZE}"
echo "[$(date)]   micro_batch=${MICRO_BATCH_SIZE}, max_len=${MAX_LENGTH}"
echo "[$(date)]   warmup=${WARMUP_RATIO}, weight_decay=${WEIGHT_DECAY}"
echo "[$(date)]   scheduler=${LR_SCHEDULER}, save_freq=${SAVE_FREQ}"
echo "[$(date)] ============================================================"

# ============================================================
# Step 0: Verify training data exists
# ============================================================
if [ ! -f "${SFT_TRAIN_PARQUET}" ]; then
    echo "[$(date)] ERROR: Training data not found: ${SFT_TRAIN_PARQUET}"
    echo "[$(date)] Please run gen_teacher_sft_data.sh first!"
    exit 1
fi

echo "[$(date)] Training data: ${SFT_TRAIN_PARQUET}"
${PYTHON} -c "
import pandas as pd
df = pd.read_parquet('${SFT_TRAIN_PARQUET}')
print(f'  Samples: {len(df)}')
print(f'  Columns: {df.columns.tolist()}')
steps_per_epoch = len(df) // ${BATCH_SIZE}
total_steps = steps_per_epoch * ${NUM_EPOCHS}
print(f'  Steps/epoch: {steps_per_epoch}')
print(f'  Total steps: {total_steps}')
print(f'  Save checkpoints at: every {${SAVE_FREQ}} steps')
"

# ============================================================
# Step 1: Delete old checkpoints
# ============================================================
echo "[$(date)] Cleaning old checkpoints..."
if [ -d "${FSDP_CKPT_DIR}" ]; then
    echo "[$(date)]   Removing old FSDP checkpoints: ${FSDP_CKPT_DIR}"
    rm -rf "${FSDP_CKPT_DIR}"
fi
if [ -d "${HF_CKPT_DIR}" ]; then
    echo "[$(date)]   Removing old HF checkpoints: ${HF_CKPT_DIR}"
    rm -rf "${HF_CKPT_DIR}"
fi
mkdir -p "${FSDP_CKPT_DIR}" "${HF_CKPT_DIR}" "${LOG_DIR}"
echo "[$(date)]   Done."

# ============================================================
# Step 2: Run SFT Training via verl's FSDPSFTTrainer
# ============================================================
echo "[$(date)] ===== Starting SFT Training ====="

/opt/conda/envs/OpenAgentRL-sj/bin/torchrun --nproc_per_node=${N_GPUS} \
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
    optim.weight_decay=${WEIGHT_DECAY} \
    optim.warmup_steps_ratio=${WARMUP_RATIO} \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=${LR_SCHEDULER} \
    trainer.total_epochs=${NUM_EPOCHS} \
    trainer.project_name=easyopd-sft \
    trainer.experiment_name=${RUN_NAME} \
    trainer.default_local_dir="${FSDP_CKPT_DIR}" \
    trainer.logger="['console']" \
    trainer.seed=${SEED} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=-1 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    hydra.run.dir="${LOG_DIR}/hydra" \
    hydra.output_subdir=null \
    hydra.job.chdir=false \
    2>&1 | tee "${LOG_DIR}/train.log"

echo "[$(date)] ===== SFT Training Completed ====="

# ============================================================
# Step 3: Merge ALL FSDP checkpoints to HuggingFace format
# ============================================================
echo "[$(date)] ===== Merging FSDP checkpoints to HuggingFace format ====="

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

echo "[$(date)] Found ${#ALL_CKPTS[@]} checkpoint(s) to merge."

for CKPT_DIR in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT_DIR}")
    TARGET_DIR="${HF_CKPT_DIR}/${STEP_NAME}"

    if [ -f "${TARGET_DIR}/model.safetensors" ] || [ -f "${TARGET_DIR}/pytorch_model.bin" ]; then
        echo "[$(date)] [${STEP_NAME}] Already merged, skipping."
        continue
    fi

    echo "[$(date)] [${STEP_NAME}] Merging ${CKPT_DIR} -> ${TARGET_DIR}"
    ${PYTHON} ${SHARED_SCRIPTS}/merge_fsdp_checkpoint.py \
        --checkpoint_dir "${CKPT_DIR}" \
        --output_dir "${TARGET_DIR}" \
        2>&1 | tee "${LOG_DIR}/merge_${STEP_NAME}.log"

    # --- Post-merge fixes ---
    # Note: With transformers 4.57.1 (unified across train/eval envs),
    # config.json is saved with correct rope_scaling field. No need to
    # copy base model's config.json anymore.

    # Fix 1: Inject chat_template from chat_template.jinja into tokenizer_config.json
    #   (verl saves chat_template.jinja but doesn't include it in tokenizer_config.json;
    #    SGLang 0.4.6 + transformers 4.46.3 need it in tokenizer_config.json)
    local_tok_config="${TARGET_DIR}/tokenizer_config.json"
    local_jinja="${TARGET_DIR}/chat_template.jinja"
    if [ -f "${local_tok_config}" ] && [ -f "${local_jinja}" ]; then
        ${PYTHON} -c "
import json
with open('${local_tok_config}') as f:
    config = json.load(f)
if 'chat_template' not in config:
    with open('${local_jinja}') as f:
        config['chat_template'] = f.read().strip()
    # Fix TokenizersBackend -> GPT2Tokenizer (transformers 4.46.3 compat)
    if config.get('tokenizer_class') == 'TokenizersBackend':
        config['tokenizer_class'] = 'GPT2Tokenizer'
    with open('${local_tok_config}', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print('  [fix] Injected chat_template + fixed tokenizer_class')
else:
    print('  [ok] chat_template already present')
"
    fi
done

echo "[$(date)] ===== Merge Completed ====="

# ============================================================
# Step 4: Verify chat template in merged checkpoints
# ============================================================
echo "[$(date)] ===== Verifying chat template (3 checks) ====="

for CKPT_DIR in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT_DIR}")
    TARGET_DIR="${HF_CKPT_DIR}/${STEP_NAME}"

    echo "[$(date)] [${STEP_NAME}] Checking chat template..."
    ${PYTHON} -c "
from transformers import AutoTokenizer
import sys

model_path = '${TARGET_DIR}'
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Check 1: chat_template is not None
assert tok.chat_template is not None, 'FAIL: chat_template is None!'
print('  Check 1 PASS: chat_template is set')

# Check 2: Correct format (Phi-4-mini uses <|role|>content<|end|>)
messages = [{'role': 'user', 'content': 'Hello'}]
result = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
assert '<|user|>' in result and '<|end|>' in result and '<|assistant|>' in result, \
    f'FAIL: Wrong template format: {result}'
print(f'  Check 2 PASS: Template format correct: {repr(result)}')

# Check 3: Full conversation format matches KDFlow
messages_full = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'What is 2+2?'},
    {'role': 'assistant', 'content': 'The answer is 4.'}
]
full_text = tok.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
assert '<|system|>' in full_text, f'FAIL: No system tag in: {full_text[:100]}'
assert '<|user|>' in full_text, f'FAIL: No user tag in: {full_text[:100]}'
assert '<|assistant|>' in full_text, f'FAIL: No assistant tag in: {full_text[:100]}'
assert '<|end|>' in full_text, f'FAIL: No end tag in: {full_text[:100]}'
# Ensure it's NOT chatml format
assert '<|im_start|>' not in full_text, f'FAIL: Contains chatml tags!'
assert '<|im_end|>' not in full_text, f'FAIL: Contains chatml tags!'
print(f'  Check 3 PASS: Full conversation format correct (not chatml)')
print(f'    Full text preview: {full_text[:150]}...')
print('  ALL CHECKS PASSED ✓')
"
done

echo "[$(date)] ===== All Verifications Passed ====="

# ============================================================
# Summary
# ============================================================
echo ""
echo "[$(date)] ============================================================"
echo "[$(date)] SFT TRAINING COMPLETE"
echo "[$(date)] ============================================================"
echo "[$(date)] Checkpoints:"
for CKPT_DIR in "${ALL_CKPTS[@]}"; do
    STEP_NAME=$(basename "${CKPT_DIR}")
    echo "[$(date)]   ${HF_CKPT_DIR}/${STEP_NAME}"
done
echo "[$(date)] ============================================================"
echo "[$(date)] Next steps:"
echo "[$(date)]   1. Run evaluation: bash eval_base.sh (or eval_all.sh)"
echo "[$(date)]   2. Select best checkpoint based on eval results"
echo "[$(date)] ============================================================"
