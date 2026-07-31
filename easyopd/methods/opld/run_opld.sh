#!/usr/bin/env bash
# OPLD: On-Policy Listwise Distillation — training launcher
#
# Teacher: Qwen3-4B-Instruct-2507   Student: Qwen3-0.6B  (defaults)
#
# Usage:
#   bash easyopd/methods/opld/run_opld.sh
#
# Dry-run (print the command, launch nothing):
#   OPLD_DRYRUN=true bash easyopd/methods/opld/run_opld.sh
#
# Common overrides (env vars):
#   MODEL_ROOT        Model directory        (default: /dockerdata/junewluo/models)
#   OPLD_STUDENT      Student model path     (default: $MODEL_ROOT/Qwen3-0.6B)
#   OPLD_TEACHER      Teacher model path     (default: $MODEL_ROOT/Qwen3-4B-Instruct-2507)
#   OPLD_TRAIN_DATA   Train parquet          (REQUIRED)
#   OPLD_VAL_DATA     Val parquet            (default: same as train)
#   OPLD_K            Rollouts per prompt    (default: 8, MUST be > 1)
#   OPLD_BETA         Softmax temperature    (default: 1.0)
#   OPLD_ETA          Task-reward weight     (default: 0.0 = pure teacher)
#   OPLD_LENGTH_NORM  Length-normalize       (default: true)
#   OPLD_KL_DIRECTION qT_to_qS | qS_to_qT    (default: qT_to_qS)
#   OPLD_STD_NORM     Std-normalize adv      (default: false)
#   N_GPUS            GPUs per node          (default: 8)
#   OPLD_LR           Learning rate          (default: 1e-6)
#
# Anything after `--` is appended verbatim as extra hydra overrides.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${OPLD_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PROJECT_ROOT

if [ "${OPLD_SKIP_REPO_DOTENV:-}" != "true" ] && [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# ---- models ----
MODEL_ROOT="${MODEL_ROOT:-/dockerdata/junewluo/models}"
STUDENT_MODEL="${OPLD_STUDENT:-$MODEL_ROOT/Qwen3-0.6B}"
TEACHER_MODEL="${OPLD_TEACHER:-$MODEL_ROOT/Qwen3-4B-Instruct-2507}"

# ---- data ----
TRAIN_DATA="${OPLD_TRAIN_DATA:-<SET_OPLD_TRAIN_DATA>}"
VAL_DATA="${OPLD_VAL_DATA:-$TRAIN_DATA}"

# ---- OPLD hyper-parameters ----
K="${OPLD_K:-8}"
BETA="${OPLD_BETA:-1.0}"
ETA="${OPLD_ETA:-0.0}"
LENGTH_NORM="${OPLD_LENGTH_NORM:-true}"
KL_DIRECTION="${OPLD_KL_DIRECTION:-qT_to_qS}"
STD_NORM="${OPLD_STD_NORM:-false}"

# ---- training ----
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"
LR="${OPLD_LR:-1e-6}"
TRAIN_BATCH_SIZE="${OPLD_TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${OPLD_PPO_MINI_BATCH_SIZE:-128}"
MAX_PROMPT_LEN="${OPLD_MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${OPLD_MAX_RESPONSE_LEN:-8192}"
PPO_MAX_TOKEN_LEN="${OPLD_PPO_MAX_TOKEN_LEN:-32768}"
ROLLOUT_TP="${OPLD_ROLLOUT_TP:-1}"
GPU_MEM_UTIL="${OPLD_GPU_MEM_UTIL:-0.85}"
TOTAL_EPOCHS="${OPLD_TOTAL_EPOCHS:-1}"

# ---- logging ----
PROJECT_NAME="${OPLD_PROJECT_NAME:-opld}"
EXPERIMENT_NAME="${OPLD_EXPERIMENT_NAME:-opld_qwen3_0.6b_from_4b}"
SAVE_DIR="${OPLD_SAVE_DIR:-./checkpoint/$EXPERIMENT_NAME}"
LOGGER="${OPLD_LOGGER:-console}"

# ---- sanity checks ----
if [ "$K" -le 1 ]; then
    echo "Error: OPLD_K must be > 1. Listwise distillation builds a softmax over the" >&2
    echo "       K rollouts of each prompt; with K=1 every group is degenerate" >&2
    echo "       (q_T = q_S = 1) and the advantage is identically zero." >&2
    exit 1
fi

HYDRA_OVERRIDES=(
    # --- method selection -------------------------------------------------
    # '+' because neither key exists in verl's ppo_trainer.yaml.
    # easyopd.method.name is REQUIRED: HookDispatcher.from_config only calls
    # ensure_discovered() when it can resolve a method name, and that discovery
    # is what imports easyopd/methods/opld and registers the `listwise`
    # advantage estimator. Without it: "Unknown advantage estimator simply: listwise".
    "+easyopd.method.name=listwise"
    "algorithm.adv_estimator=listwise"

    # --- OPLD hyper-parameters -------------------------------------------
    # Under algorithm.easyopd (a declared AlgoConfig field). A top-level
    # algorithm.listwise_kl would be silently swallowed by BaseConfig.get.
    "+algorithm.easyopd.listwise.beta=$BETA"
    "+algorithm.easyopd.listwise.eta=$ETA"
    "+algorithm.easyopd.listwise.length_norm=$LENGTH_NORM"
    "+algorithm.easyopd.listwise.kl_direction=$KL_DIRECTION"
    "+algorithm.easyopd.listwise.std_norm=$STD_NORM"

    # --- teacher as the reference worker ----------------------------------
    # OPLD needs teacher per-token logprobs on the batch BEFORE compute_advantage
    # (ray_trainer.py:2484). Pointing the ref worker at the teacher makes the frozen
    # "reference" forward the teacher scoring the student's rollouts; the result
    # lands as `ref_log_prob` at ray_trainer.py:2392, comfortably earlier.
    # '+' because ref.yaml declares no `model:` block -- fsdp_workers.py:673-676
    # reads it optionally and falls back to the actor path when absent.
    "+actor_rollout_ref.ref.model.path=$TEACHER_MODEL"

    # main_ppo.add_ref_policy_worker (main_ppo.py:204-213) only spawns Role.RefPolicy
    # when one of use_kl_in_reward / use_kl_loss / token_kl_reg.enable / opsa_enable
    # is set. OPLD needs none of them semantically -- our advantage IS already the
    # gradient of the listwise KL, so no extra KL term belongs anywhere. We flip the
    # one switch that can be made completely inert, purely to get the ref worker:
    #
    #   stepwise_enable=False  -> skips the SOD branch (ray_trainer.py:1640)
    #   beta_max=null          -> legacy branch returns the batch untouched
    #                             at ray_trainer.py:1717-1718, before it can
    #                             touch batch["advantages"]
    #   beta_min=0.0, gamma=1.0, coef=0.0
    #                          -> pinned so the guard `beta_max <= beta_min`
    #                             also holds if someone later sets beta_max.
    #                             (coef is a declared TokenKLRegConfig field that
    #                             no code path actually reads -- pinned for clarity,
    #                             not because it does anything.)
    #
    # Net effect: the ref worker spawns; advantages are not modified.
    # '+' because token_kl_reg is an AlgoConfig dataclass field with no yaml entry.
    "+algorithm.token_kl_reg.enable=True"
    "+algorithm.token_kl_reg.stepwise_enable=False"
    "+algorithm.token_kl_reg.beta_max=null"
    "+algorithm.token_kl_reg.beta_min=0.0"
    "+algorithm.token_kl_reg.coef=0.0"
    "+algorithm.token_kl_reg.gamma=1.0"
    "algorithm.use_kl_in_reward=False"
    "actor_rollout_ref.actor.use_kl_loss=False"

    # --- data -------------------------------------------------------------
    "data.train_files=['$TRAIN_DATA']"
    "data.val_files=['$VAL_DATA']"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LEN"
    "data.max_response_length=$MAX_RESPONSE_LEN"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.apply_chat_template_kwargs.enable_thinking=False"

    # --- student model / actor -------------------------------------------
    "actor_rollout_ref.model.path=$STUDENT_MODEL"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.optim.lr=$LR"
    "actor_rollout_ref.actor.optim.warmup_style=constant"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"

    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.65"
    "actor_rollout_ref.rollout.max_num_batched_tokens=16384"
    # "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=true"
    "actor_rollout_ref.rollout.val_kwargs.temperature=1.0"
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
    "actor_rollout_ref.rollout.val_kwargs.n=8"

    # --- rollout: K candidates per prompt is the whole point --------------
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=$K"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL"

    # --- no critic: the listwise advantage feeds the PG path directly -----
    "critic.enable=False"

    # --- trainer ----------------------------------------------------------
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.default_local_dir=$SAVE_DIR"
    "trainer.logger=[$LOGGER]"
    "trainer.n_gpus_per_node=$N_GPUS"
    "trainer.nnodes=$NNODES"
    "trainer.val_before_train=true"
    "trainer.test_freq=14"
    "trainer.save_freq=-1"
    "trainer.total_epochs=$TOTAL_EPOCHS"
)

if [ "${OPLD_DRYRUN:-}" = "true" ]; then
    set +x
    echo ""
    echo "=== OPLD Dry-Run ==="
    echo "PROJECT_ROOT:   $PROJECT_ROOT"
    echo "student:        $STUDENT_MODEL"
    echo "teacher (ref):  $TEACHER_MODEL"
    echo "train data:     $TRAIN_DATA"
    echo "K (rollout.n):  $K"
    echo "beta/eta:       $BETA / $ETA"
    echo "length_norm:    $LENGTH_NORM"
    echo "kl_direction:   $KL_DIRECTION"
    echo "GPUs:           ${N_GPUS} x ${NNODES} node(s)"
    echo ""
    echo "python3 -m verl.trainer.main_ppo \\"
    for override in "${HYDRA_OVERRIDES[@]}"; do
        echo "    $override \\"
    done
    echo "    $*"
    exit 0
fi

if [[ "$TRAIN_DATA" == *"<"* ]]; then
    echo "Error: set OPLD_TRAIN_DATA to a training parquet." >&2
    echo "       Any GRPO-style parquet works (prompt + reward_model columns);" >&2
    echo "       OPLD's supervision comes from the teacher, not the reward." >&2
    exit 1
fi

for path in "$STUDENT_MODEL" "$TEACHER_MODEL"; do
    if [ ! -d "$path" ]; then
        echo "Error: model directory not found: $path" >&2
        exit 1
    fi
done

python3 -m verl.trainer.main_ppo "${HYDRA_OVERRIDES[@]}" "$@"
