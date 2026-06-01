# lightning_opd Trainer

Offline on-policy distillation with precomputed teacher log-probabilities.

**Paper**: [arXiv:2604.13010](https://arxiv.org/abs/2604.13010)
**Source**: [NVIDIA-NeMo/Lightning-OPD](https://github.com/NVIDIA-NeMo/Lightning-OPD)

## Quick Start

### Dry-run (no GPU required)

```bash
# Training entrypoint dry-run
LIGHTNING_OPD_DRYRUN=true LIGHTNING_OPD_SKIP_REPO_DOTENV=true \
    bash examples/lightning_opd_trainer/train_lightning_opd.sh

# Data preparation dry-run
LIGHTNING_OPD_DRYRUN=true bash examples/lightning_opd_trainer/tools/prepare_data.sh
```

### Full pipeline (Step 0–6)

```bash
# Step 0: Prepare SFT prompts
LIGHTNING_OPD_HF_DATASET=open-thoughts/OpenThoughts3-1.2M \
    bash examples/lightning_opd_trainer/tools/prepare_sft_prompts.sh

# Step 1: Generate SFT data (requires teacher model served by OpenAI/vLLM endpoint)
LIGHTNING_OPD_TEACHER_MODEL=/path/to/teacher \
    LIGHTNING_OPD_TEACHER_CHAT_URL=http://127.0.0.1:8000/v1/chat/completions \
    LIGHTNING_OPD_SFT_PROMPTS=./data/prompts/openthoughts3_300k.jsonl \
    bash examples/lightning_opd_trainer/tools/generate_sft_data.sh

# Step 2: SFT training
LIGHTNING_OPD_SFT_BASE_MODEL=/path/to/base_model \
    LIGHTNING_OPD_SFT_DATA=./data/sft_generated/train.parquet \
    bash examples/lightning_opd_trainer/tools/run_sft.sh

# Step 3: Collect student rollouts
LIGHTNING_OPD_SFT_CHECKPOINT=./checkpoint/sft_lightning_opd \
    LIGHTNING_OPD_STUDENT_URL=http://127.0.0.1:8000/v1/chat/completions \
    LIGHTNING_OPD_OPD_PROMPTS=./data/prompts/dapo_math_17k.jsonl \
    bash examples/lightning_opd_trainer/tools/collect_rollouts.sh

# Step 4: Precompute teacher logprobs
LIGHTNING_OPD_TOKENIZER=./checkpoint/sft_lightning_opd \
    LIGHTNING_OPD_ROLLOUTS=./data/rollouts/rollouts.parquet \
    LIGHTNING_OPD_TEACHER_URL=http://127.0.0.1:8000/v1/completions \
    bash examples/lightning_opd_trainer/tools/prepare_data.sh

# Step 5: lightning_opd training
LIGHTNING_OPD_SFT_CHECKPOINT=./checkpoint/sft_lightning_opd \
    LIGHTNING_OPD_DATA=./data/lightning_opd/rollouts-lightning_opd-precomputed.parquet \
    MODEL_SCALE=4b \
    bash examples/lightning_opd_trainer/train_lightning_opd.sh

# Step 6: Megatron→HF conversion (if using Megatron backend)
bash examples/lightning_opd_trainer/tools/convert_megatron_to_hf.sh
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LIGHTNING_OPD_PROJECT_ROOT` | Project root override | auto-detect |
| `LIGHTNING_OPD_DRYRUN` | Print commands without executing | `false` |
| `LIGHTNING_OPD_SFT_CHECKPOINT` | SFT checkpoint path | required |
| `LIGHTNING_OPD_DATA` | Precomputed parquet path | required |
| `LIGHTNING_OPD_TEACHER_MODEL` | Teacher model path | optional |
| `LIGHTNING_OPD_TEACHER_URL` | Teacher logprob completions URL (Step 4) | `http://127.0.0.1:8000/v1/completions` |
| `LIGHTNING_OPD_TEACHER_CHAT_URL` | Teacher generation chat URL (Step 1) | `http://127.0.0.1:8000/v1/chat/completions` |
| `LIGHTNING_OPD_STUDENT_URL` | SFT student OpenAI/vLLM URL | `http://127.0.0.1:8000/v1/chat/completions` |
| `MODEL_SCALE` | Model scale: `4b`, `8b` | `4b` |

## Recommended Directory Layout

```
data/
  prompts/           # Step 0 output: JSONL prompt files
  sft_generated/     # Step 1 output: teacher-generated SFT data
  rollouts/          # Step 3 output: student rollouts
  lightning_opd/     # Step 4 output: precomputed parquet

checkpoint/
  sft_lightning_opd/ # Step 2 output: SFT checkpoint
  lightning_opd_4b/  # Step 5 output: lightning_opd checkpoint
```

## Not Migrated (by design)

- `slime/` framework: not needed; EasyOPD uses verl
- `configs/lightning_opd/*.py` — translated to Hydra YAML configs
- `configs/models/*.sh` — model architecture handled by verl model loader
- `configs/sft/*.yaml` (LlamaFactory) — translated to `easyopd/config/lightning_opd/sft.yaml`
- `train.py` — replaced by `verl.trainer.main_ppo`
- `run_docker.sh` — uses EasyOPD's existing docker setup
- `assets/` — referenced via external links in docs

## Model Scale

- **4B** (Qwen3-4B student, Qwen3-8B teacher): Verified path
- **8B** (Qwen3-8B student, Qwen3-32B teacher): Verified path
- **30B-A3B** (MoE): Stretch goal; not yet end-to-end verified in EasyOPD
