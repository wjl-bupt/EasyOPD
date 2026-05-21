# SOD: Step-wise On-policy Distillation

## Method Overview

SOD addresses the instability of standard OPD in tool-integrated reasoning (TIR) scenarios.
It adaptively re-weights the distillation strength at each reasoning step based on the divergence trajectory.

**Paper:** https://arxiv.org/abs/2605.07725

---

## Environment Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | >= 3.10 | Tested with 3.11 |
| CUDA | >= 12.1 | Tested with 12.4 |
| PyTorch | >= 2.4 | |
| vLLM | 0.8.4-0.8.5 | Tested with 0.8.5.post1 |
| flash-attn | >= 2.5 | |
| Ray | >= 2.10 | For distributed training |
| verl | From source | pip install -e . from EasyOPD root |

### Hardware Requirements

- **8x NVIDIA H20 96GB GPUs**
- batch_size=64, n=16 (1024 rollout samples per step)
- infer_tp=4, train_sp=4
- Peak GPU memory: ~88 GB per GPU

---

## Quick Start

### 1. Environment Setup

```bash
# Activate the conda environment with vLLM 0.8.5
conda activate OpenAgentRL

# Enable internet proxy (if needed for sandbox access)
source /path/to/enable_internet_proxy.sh

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1
```

### 2. Install EasyOPD

```bash
cd /path/to/EasyOPD
pip install -e .
```

### 3. Run Training

```bash
cd /path/to/EasyOPD
bash examples/sod/run_sod.sh
```

Or run directly with custom parameters (see examples/sod/run_sod.sh for full command).

---

## File Structure

```
EasyOPD/
 easyopd/methods/sod/
   ├── __init__.py          # Exports and metadata
   ├── core.py              # Core SOD algorithm (stepwise KL regularizer)
   └── README.md            # This file
 easyopd/config/sod.yaml  # Config template
 examples/sod/run_sod.sh  # Training launch script
 recipe/demystify/
   ├── reward.py            # Reward function with sandbox code execution
   └── sandbox_fusion_tool_config.yaml  # Sandbox URL and tool config
 verl/
    ├── trainer/config/algorithm.py      # TokenKLRegConfig dataclass
    ├── trainer/ppo/ray_trainer.py       # _apply_token_kl_regularizer()
    ├── tools/sandbox_fusion_tools.py    # Sandbox execution tool
    └── utils/reward_score/livecodebench/  # LiveCodeBench evaluation
```

---

## Configuration Details

### SOD-specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `token_kl_reg.enable` | True | Enable token-level KL regularization |
| `token_kl_reg.gamma` | 1.0 | Discount factor for KL penalty |
| `token_kl_reg.beta_min` | 0.0 | Minimum KL coefficient |
| `token_kl_reg.beta_max` | 0.10 | Maximum KL coefficient |
| `token_kl_reg.stepwise_enable` | True | Enable step-wise adaptive weighting |
| `token_kl_reg.stepwise_epsilon` | 1e-6 | Numerical stability epsilon |
| `token_kl_reg.stepwise_delta` | 0.2 | Divergence threshold for step weighting |
| `token_kl_reg.stepwise_opd_coef` | 1.0 | OPD loss coefficient |

### Sandbox Configuration

Edit `recipe/demystify/sandbox_fusion_tool_config.yaml` to set your sandbox URL:

```yaml
tools:
  - class_name: "recipe.demystify.reward.CustomSandboxFusionTool"
    config:
      sandbox_fusion_url: "YOUR_SANDBOX_URL_HERE"
      num_workers: 32
      enable_global_rate_limit: true
      rate_limit: 128
      default_timeout: 60
      default_language: "python"
```

---

## Validated Results

Successfully validated with the following configuration:

- **Environment**: conda env OpenAgentRL (Python 3.11, vLLM 0.8.5.post1)
- **Hardware**: 8x NVIDIA H20 96GB GPUs
- **Student Model**: Qwen3-1.7B-SFT
- **Teacher Model**: DemyAgent-4B
- **Dataset**: Open-AgentRL-30K

### Step 1 Metrics (Verified):

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

### 1. GPU OOM (No available memory for cache blocks)
**Error**: `ValueError: No available memory for the cache blocks`
**Cause**: Other processes occupying GPU memory
**Solution**: Ensure all GPUs are free before starting training. Check with `nvidia-smi`.

### 2. WandB API Key
**Error**: `wandb.errors.errors.UsageError: No API key`
**Solution**: Use `trainer.logger=['console']` to disable wandb, or set `WANDB_API_KEY` environment variable.

### 3. Sandbox API Errors (500)
**Error**: `API Request Error: 500 Server Error`
**Cause**: Sandbox service temporary unavailability
**Solution**: These are transient errors and won't crash training. The reward function handles them gracefully.

### 4. FlashInfer Warning
**Warning**: `FlashInfer is not available. Falling back to PyTorch-native implementation`
**Impact**: Slightly slower sampling, does not affect correctness.
**Solution**: Install FlashInfer for better performance (optional).
