#!/bin/bash
set -euo pipefail
export PYTHONPATH="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
PYTHON="/opt/conda/envs/OpenAgentRL-sj/bin/python"
RAY="/opt/conda/envs/OpenAgentRL-sj/bin/ray"
export PATH="/opt/conda/envs/OpenAgentRL-sj/bin:$PATH"
BENCH_DIR="/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/benchmark"
TEACHER_MODEL="/root/workspace/models/Qwen2.5-7B-Instruct"
STUDENT_BASE="/root/workspace/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6/checkpoint-40"
TEACHER_SFT_DIR="${BENCH_DIR}/checkpoints/teacher_sft_phi4mini"
TEACHER_SFT_MODEL="${TEACHER_SFT_DIR}/final"
SFT_DATA_DIR="${BENCH_DIR}/teacher_sft_data"
LOCAL_MODEL_DIR="/root/workspace/models"

echo "[$(date)] ===== STEP 1: Generate teacher responses ====="
if [ ! -f "${SFT_DATA_DIR}/teacher_sft_train.jsonl" ]; then
    ${PYTHON} ${BENCH_DIR}/gen_teacher.py --teacher_model ${TEACHER_MODEL} --tp 4
    echo "[$(date)] Teacher responses generated."
else
    echo "[$(date)] Teacher responses already exist, skipping."
fi

echo "[$(date)] ===== STEP 2: SFT student on teacher responses ====="
if [ ! -d "${TEACHER_SFT_MODEL}" ]; then
    /opt/conda/envs/OpenAgentRL-sj/bin/accelerate launch \
        --num_processes=8 \
        --mixed_precision=bf16 \
        ${BENCH_DIR}/run_teacher_sft.py \
        --student_model ${STUDENT_BASE} \
        --data_path ${SFT_DATA_DIR}/teacher_sft_train.jsonl \
        --output_dir ${TEACHER_SFT_DIR} \
        --num_epochs 3 --lr 2e-5 --batch_size 2 --grad_accum 8
    echo "[$(date)] Teacher SFT completed."
else
    echo "[$(date)] Teacher SFT model already exists, skipping."
fi

echo "[$(date)] ===== STEP 3: Evaluate Teacher SFT baseline ====="
LOCAL_TEACHER_SFT="${LOCAL_MODEL_DIR}/teacher_sft_phi4mini"
if [ ! -f "${LOCAL_TEACHER_SFT}/config.json" ]; then
    rm -rf ${LOCAL_TEACHER_SFT}
    echo "[$(date)] Copying Teacher SFT model to local fast storage..."
    cp -r ${TEACHER_SFT_MODEL} ${LOCAL_TEACHER_SFT}
    echo "[$(date)] Copy done."
fi
# Ensure tokenizer files from base model are present
if [ ! -f "${LOCAL_TEACHER_SFT}/vocab.json" ]; then
    cp ${STUDENT_BASE}/tokenizer_config.json ${LOCAL_TEACHER_SFT}/tokenizer_config.json
    cp ${STUDENT_BASE}/vocab.json ${LOCAL_TEACHER_SFT}/vocab.json
    cp ${STUDENT_BASE}/merges.txt ${LOCAL_TEACHER_SFT}/merges.txt
    echo "[$(date)] Tokenizer files patched."
fi
${PYTHON} ${BENCH_DIR}/evaluate_model.py --model_path ${LOCAL_TEACHER_SFT} --model_name "Teacher_SFT_Phi4mini" --tensor_parallel_size 1 --dp_size 8 --benchmarks math500
${PYTHON} ${BENCH_DIR}/evaluate_model.py --model_path ${LOCAL_TEACHER_SFT} --model_name "Teacher_SFT_Phi4mini" --tensor_parallel_size 1 --dp_size 8 --benchmarks gsm8k

echo "[$(date)] ===== STEP 4: Run Simple on Teacher SFT model ====="
export RAY_ADDRESS=auto
${RAY} stop --force 2>/dev/null || true
sleep 3
${RAY} start --head --disable-usage-stats --num-cpus=64 --num-gpus=8 --include-dashboard=false
SIMPLE_CKPT="${BENCH_DIR}/checkpoints/simple_on_teacher_sft"
mkdir -p ${SIMPLE_CKPT}
${PYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=True \
    "data.train_files=['${SFT_DATA_DIR}/train_for_rl.parquet']" \
    "data.val_files=['${BENCH_DIR}/data_phi4mini/val.parquet']" \
    data.train_batch_size=16 data.max_prompt_length=512 data.max_response_length=1024 \
    data.filter_overlong_prompts=True data.truncation=right data.prompt_key=prompt \
    actor_rollout_ref.model.path="${LOCAL_TEACHER_SFT}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    "++actor_rollout_ref.actor.policy_loss.loss_mode=simple"    "++actor_rollout_ref.actor.policy_loss.simple_kl_direction=reverse" \
    "++actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_model_len=1537 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${BENCH_DIR}/reward_fn.py" \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True 'trainer.logger=["console"]' \
    trainer.project_name=easyopd_benchmark trainer.experiment_name=simple_on_tsft \
    trainer.n_gpus_per_node=8 trainer.nnodes=1 \
    trainer.val_before_train=False trainer.total_training_steps=200 \
    trainer.save_freq=200 trainer.test_freq=50 \
    trainer.default_local_dir="${SIMPLE_CKPT}"
echo "[$(date)] Simple training completed."

echo "[$(date)] ===== STEP 5: Merge Simple checkpoint ====="
SIMPLE_ACTOR_CKPT=$(find ${SIMPLE_CKPT} -name "actor" -type d | head -1)
${PYTHON} ${BENCH_DIR}/merge_fsdp.py --ckpt_dir ${SIMPLE_ACTOR_CKPT} --base_model ${TEACHER_SFT_MODEL} --output_dir ${SIMPLE_CKPT}/merged_hf

echo "[$(date)] ===== STEP 6: Evaluate Simple ====="
${RAY} stop --force 2>/dev/null || true
LOCAL_SIMPLE="${LOCAL_MODEL_DIR}/simple_on_teacher_sft"
rm -rf ${LOCAL_SIMPLE}
echo "[$(date)] Copying Simple model to local fast storage..."
cp -r ${SIMPLE_CKPT}/merged_hf ${LOCAL_SIMPLE}
# Ensure tokenizer files from base model are present
if [ ! -f "${LOCAL_SIMPLE}/vocab.json" ]; then
    cp ${STUDENT_BASE}/tokenizer_config.json ${LOCAL_SIMPLE}/tokenizer_config.json
    cp ${STUDENT_BASE}/vocab.json ${LOCAL_SIMPLE}/vocab.json
    cp ${STUDENT_BASE}/merges.txt ${LOCAL_SIMPLE}/merges.txt
fi
echo "[$(date)] Copy done."
${PYTHON} ${BENCH_DIR}/evaluate_model.py --model_path ${LOCAL_SIMPLE} --model_name "Simple_on_TeacherSFT" --tensor_parallel_size 1 --dp_size 8 --benchmarks math500
${PYTHON} ${BENCH_DIR}/evaluate_model.py --model_path ${LOCAL_SIMPLE} --model_name "Simple_on_TeacherSFT" --tensor_parallel_size 1 --dp_size 8 --benchmarks gsm8k

echo "[$(date)] ===== STEP 7: Run SimCT on Teacher SFT model ====="
${RAY} stop --force 2>/dev/null || true
sleep 3
${RAY} start --head --disable-usage-stats --num-cpus=64 --num-gpus=8 --include-dashboard=false
SIMCT_CKPT="${BENCH_DIR}/checkpoints/simct_on_teacher_sft"
mkdir -p ${SIMCT_CKPT}
${PYTHON} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=True \
    "data.train_files=['${SFT_DATA_DIR}/train_for_rl.parquet']" \
    "data.val_files=['${BENCH_DIR}/data_phi4mini/val.parquet']" \
    data.train_batch_size=16 data.max_prompt_length=512 data.max_response_length=1024 \
    data.filter_overlong_prompts=True data.truncation=right data.prompt_key=prompt \
    actor_rollout_ref.model.path="${LOCAL_TEACHER_SFT}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    "++actor_rollout_ref.actor.policy_loss.loss_mode=simct"    "++actor_rollout_ref.actor.policy_loss.simple_kl_direction=reverse" \
    "++actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_model_len=1537 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${BENCH_DIR}/reward_fn.py" \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True 'trainer.logger=["console"]' \
    trainer.project_name=easyopd_benchmark trainer.experiment_name=simct_on_tsft \
    trainer.n_gpus_per_node=8 trainer.nnodes=1 \
    trainer.val_before_train=False trainer.total_training_steps=200 \
    trainer.save_freq=200 trainer.test_freq=50 \
    trainer.default_local_dir="${SIMCT_CKPT}"
echo "[$(date)] SimCT training completed."

echo "[$(date)] ===== STEP 8: Merge SimCT checkpoint ====="
SIMCT_ACTOR_CKPT=$(find ${SIMCT_CKPT} -name "actor" -type d | head -1)
${PYTHON} ${BENCH_DIR}/merge_fsdp.py --ckpt_dir ${SIMCT_ACTOR_CKPT} --base_model ${TEACHER_SFT_MODEL} --output_dir ${SIMCT_CKPT}/merged_hf

echo "[$(date)] ===== STEP 9: Evaluate SimCT ====="
${RAY} stop --force 2>/dev/null || true
LOCAL_SIMCT="${LOCAL_MODEL_DIR}/simct_on_teacher_sft"
rm -rf ${LOCAL_SIMCT}
echo "[$(date)] Copying SimCT model to local fast storage..."
cp -r ${SIMCT_CKPT}/merged_hf ${LOCAL_SIMCT}
# Ensure tokenizer files from base model are present
if [ ! -f "${LOCAL_SIMCT}/vocab.json" ]; then
    cp ${STUDENT_BASE}/tokenizer_config.json ${LOCAL_SIMCT}/tokenizer_config.json
    cp ${STUDENT_BASE}/vocab.json ${LOCAL_SIMCT}/vocab.json
    cp ${STUDENT_BASE}/merges.txt ${LOCAL_SIMCT}/merges.txt
fi
echo "[$(date)] Copy done."
${PYTHON} ${BENCH_DIR}/evaluate_model.py --model_path ${LOCAL_SIMCT} --model_name "SimCT_on_TeacherSFT" --tensor_parallel_size 1 --dp_size 8 --benchmarks math500
${PYTHON} ${BENCH_DIR}/evaluate_model.py --model_path ${LOCAL_SIMCT} --model_name "SimCT_on_TeacherSFT" --tensor_parallel_size 1 --dp_size 8 --benchmarks gsm8k

echo "[$(date)] ===== ALL DONE ====="
