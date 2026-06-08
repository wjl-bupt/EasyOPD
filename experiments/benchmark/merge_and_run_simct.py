"""Merge FSDP shards for Simple model, then start SimCT training."""
import torch
import os, glob, subprocess, sys

CKPT_DIR = '/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/benchmark/checkpoints/simple_phi4mini/global_step_200/actor'
OUTPUT_DIR = '/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/benchmark/checkpoints/simple_phi4mini/merged_hf'
BASE_MODEL = '/root/workspace/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6/checkpoint-40'

print(f"[MERGE] Loading FSDP shards from {CKPT_DIR}...")
shard_files = sorted(glob.glob(os.path.join(CKPT_DIR, 'model_world_size_8_rank_*.pt')))
print(f"[MERGE] Found {len(shard_files)} shard files")

state_dict = {}
for f in shard_files:
    print(f"[MERGE] Loading {os.path.basename(f)}...")
    shard = torch.load(f, map_location='cpu')
    state_dict.update(shard)
    del shard

print(f"[MERGE] Total parameters: {len(state_dict)}")

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"[MERGE] Loading base model from {BASE_MODEL}...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("[MERGE] Applying state dict...")
model.load_state_dict(state_dict, strict=True)
del state_dict

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[MERGE] Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
del model
print("[MERGE] Simple model merge DONE!")

# Now start SimCT training
print("[SIMCT] Restarting Ray and launching SimCT training...")
os.system("/opt/conda/envs/OpenAgentRL-sj/bin/ray stop --force 2>/dev/null")
import time
time.sleep(5)
os.system("/opt/conda/envs/OpenAgentRL-sj/bin/ray start --head --disable-usage-stats --num-cpus=64 --num-gpus=8 --include-dashboard=false")
time.sleep(10)

# Launch SimCT
WORKDIR = "/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD"
os.chdir(WORKDIR)
os.environ["PYTHONPATH"] = f"{WORKDIR}:{os.environ.get('PYTHONPATH', '')}"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["RAY_ADDRESS"] = "auto"

SIMCT_CKPT = f"{WORKDIR}/experiments/benchmark/checkpoints/simct_phi4mini"
os.makedirs(SIMCT_CKPT, exist_ok=True)

cmd = f"""
/opt/conda/envs/OpenAgentRL-sj/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    "data.train_files=['{WORKDIR}/experiments/benchmark/data_phi4mini/train.parquet']" \
    "data.val_files=['{WORKDIR}/experiments/benchmark/data_phi4mini/val.parquet']" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path={BASE_MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    "++actor_rollout_ref.actor.policy_loss.loss_mode=simct" \
    "++actor_rollout_ref.actor.policy_loss.simple_kl_direction=reverse" \
    "++actor_rollout_ref.actor.policy_loss.simple_loss_clamp=10.0" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_model_len=1537 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path={WORKDIR}/experiments/benchmark/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    'trainer.logger=["console"]' \
    trainer.project_name=easyopd_benchmark \
    trainer.experiment_name=simct_phi4mini \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=200 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.default_local_dir={SIMCT_CKPT}
"""

print("[SIMCT] Starting SimCT training...")
ret = os.system(cmd)
print(f"[SIMCT] Training finished with return code: {ret}")
