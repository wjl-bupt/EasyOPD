# SOD: Step-wise On-policy Distillation for Small Language Model Agents

## Method Overview

SOD addresses the instability of standard OPD in tool-integrated reasoning (TIR) scenarios.
It adaptively re-weights the distillation strength at each reasoning step based on the divergence trajectory, preventing harmful supervision when cascade failures occur.

**Paper:** https://arxiv.org/abs/2605.07725
**Code:** https://github.com/YoungZ365/SOD/tree/main

---

## Environment Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | >= 3.10 | Tested with 3.11 |
| CUDA | >= 12.1 | Tested with 12.4 |
| PyTorch | >= 2.4 | |
| vLLM | 0.8.4 ~ 0.8.5 | **Must be 0.8.x** (tested with 0.8.5.post1) |
| flash-attn | >= 2.5 | |
| Ray | >= 2.10 | For distributed training |
| verl | From source | `pip install -e .` from EasyOPD root |

> **Important**: The verl framework bundled in EasyOPD is designed for vLLM 0.8.x. Do NOT use vLLM 0.9+ as the rollout API is incompatible.

### Hardware Requirements

- **8x NVIDIA H20 96GB GPUs** (or equivalent with >=88GB per GPU)
- batch_size=64, n=16 -> 1024 rollout samples per step
- infer_tp=4, train_sp=4
- Peak GPU memory: ~88 GB per GPU
- **All 8 GPUs must be free** before starting training (no other processes occupying GPU memory)

---

## Quick Start

### Step 1: Install EasyOPD

```bash
cd /path/to/EasyOPD
pip install -e .
```

### Step 2: Set Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1   # Required: enables vLLM V1 engine for async rollout
export SANDBOX_FUSION_URL="https://your-sandbox-service.com/run_code"  # Your sandbox API endpoint
```

> **Critical**: `VLLM_USE_V1=1` must be set, otherwise the async vLLM server will fail to initialize.
>
> **Critical**: `SANDBOX_FUSION_URL` must point to a valid code execution sandbox. This single environment variable is used by both the rollout tool config and the LiveCodeBench evaluation module (`code_math.py`), so you only need to set it once.

### Step 3: Configure Sandbox URL

The sandbox URL is configured via the `SANDBOX_FUSION_URL` environment variable (set in Step 2). It is used in **3 places**:

1. `examples/sod/sandbox_fusion_tool_config.yaml` — multi-turn code execution during rollout
2. `verl/utils/reward_score/livecodebench/code_math.py` (line 225) — LiveCodeBench evaluation
3. `verl/utils/reward_score/livecodebench/code_math.py` (line 291) — LiveCodeBench evaluation

You also need to edit `examples/sod/sandbox_fusion_tool_config.yaml` and replace `<YOUR_SANDBOX_URL>` with the same URL:

```yaml
tools:
  - class_name: "examples.sod.reward.CustomSandboxFusionTool"
    config:
      sandbox_fusion_url: "https://your-sandbox-service.com/run_code"
      ...
```

> **Note**: The `code_math.py` entries read from `SANDBOX_FUSION_URL` env var automatically. The yaml config file must be edited manually since it is loaded by the tool registry at runtime.

### Step 4: Configure Paths in `run_sod.sh`

Edit `examples/sod/run_sod.sh` and set the following variables:

```bash
STUDENT_MODEL_PATH="/path/to/student_model"   # HuggingFace format SFT checkpoint
TEACHER_MODEL_PATH="/path/to/teacher_model"   # HuggingFace format teacher model
TRAIN_DATA="/path/to/Open-AgentRL-30K.parquet"
VAL_DATA_1="/path/to/aime_2025_problems.parquet"
VAL_DATA_2="/path/to/aime_2024_problems.parquet"
```

### Step 5: Run Training

```bash
cd /path/to/EasyOPD
bash examples/sod/run_sod.sh
```

> **Note**: Ensure no other GPU processes are running. Use `nvidia-smi` to verify all GPUs show 0 MiB usage before starting.

---

## File Structure

```
EasyOPD/
|-- easyopd/methods/sod/
|   |-- __init__.py          # Exports and SODMethod metadata class
|   |-- core.py              # Core SOD algorithm (stepwise KL regularizer)
|   +-- README.md            # This file
|-- easyopd/config/sod.yaml  # Config template (reference only)
|-- examples/sod/run_sod.sh  # Training launch script (edit paths here)
|-- examples/sod/
|   |-- reward.py            # Reward function with sandbox code execution
|   +-- sandbox_fusion_tool_config.yaml  # Sandbox URL config (edit URL here)
+-- verl/
    |-- trainer/config/algorithm.py      # TokenKLRegConfig dataclass
    |-- trainer/ppo/ray_trainer.py       # _apply_token_kl_regularizer()
    |-- tools/sandbox_fusion_tools.py    # Sandbox execution tool
    +-- utils/reward_score/livecodebench/  # LiveCodeBench evaluation
```

---

## SOD Configuration Parameters

These parameters are passed via command line (Hydra overrides) in `run_sod.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `token_kl_reg.enable` | True | Enable token-level KL regularization |
| `token_kl_reg.gamma` | 1.0 | Discount factor for KL penalty |
| `token_kl_reg.beta_min` | 0.0 | Minimum KL coefficient |
| `token_kl_reg.beta_max` | 0.10 | Maximum KL coefficient |
| `token_kl_reg.stepwise_enable` | True | Enable step-wise adaptive weighting (core SOD feature) |
| `token_kl_reg.stepwise_epsilon` | 1e-6 | Numerical stability epsilon (Eq. 7) |
| `token_kl_reg.stepwise_delta` | 0.2 | Divergence threshold for step weighting (Eq. 7) |
| `token_kl_reg.stepwise_opd_coef` | 1.0 | OPD loss coefficient (Eq. 10) |

### Other Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data.filter_overlong_prompts` | True | Filter prompts exceeding max_prompt_length |
| `data.max_prompt_length` | 2560 | Maximum prompt token length |
| `data.max_response_length` | 20480 | Maximum response token length (multi-turn) |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.75 | vLLM GPU memory fraction |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 4 | TP size for vLLM inference |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | 4 | SP size for training |
| `trainer.logger` | `["console"]` | Use `["console","wandb"]` if WANDB_API_KEY is set |

---

## Validated Results

Successfully validated with the following configuration:

- **Environment**: Python 3.11, vLLM 0.8.5.post1, PyTorch 2.4+
- **Hardware**: 8x NVIDIA H20 96GB GPUs
- **Student Model**: Qwen3-1.7B-SFT
- **Teacher Model**: DemyAgent-4B (Gen-Verse)
- **Dataset**: Open-AgentRL-30K (training), AIME 2024/2025 (validation)

### Step 1 Metrics:

| Metric | Value |
|--------|-------|
| actor/token_kl/mean | -0.314 |
| actor/pg_loss | 0.010 |
| actor/entropy | 0.435 |
| critic/score/mean | -0.572 |
| rollout/tool_calls_avg | 3.21 |
| rollout/tool_success_rate | 74.5% |
| num_turns/mean | 8.42 |
| timing_s/step | ~1383s (~23min) |
| perf/throughput | 458 tokens/s |
| perf/max_memory_allocated_gb | 88.2 |

---

## Troubleshooting

### GPU OOM: "No available memory for the cache blocks"

This error means other processes are occupying GPU memory. Before starting training:
```bash
nvidia-smi  # Verify all GPUs show 0 MiB usage
pkill -f "main_ppo"  # Kill any leftover training processes
ray stop --force      # Clean up Ray cluster
```

### WandB API Key Error

If you see `wandb.errors.errors.UsageError: No API key configured`, either:
- Set `trainer.logger=["console"]` in the training command (already default in `run_sod.sh`)
- Or configure: `export WANDB_API_KEY=your_key`

### Sandbox 500 Errors During Training

Occasional `API Request Error: 500 Server Error` from the sandbox service are **normal** and will not crash training. The reward function handles these gracefully by assigning zero reward to failed executions.

### FlashInfer Warning

`FlashInfer is not available` is a non-critical warning. Training will use PyTorch-native attention instead, with slightly lower throughput but identical results.

### Training Takes Long to Start

Model loading from shared filesystem can be slow (~5-10 minutes). The first training step also takes longer due to multi-turn rollout with sandbox interaction (1024 samples x multiple turns x network latency). Expect ~23 minutes per step after initialization.
