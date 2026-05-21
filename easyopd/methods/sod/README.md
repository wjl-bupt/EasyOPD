# SOD: Step-wise On-policy Distillation

## Method Overview

SOD (Step-wise On-policy Distillation) addresses the instability of standard OPD in tool-integrated reasoning (TIR) scenarios. When a student model makes erroneous tool calls, cascade failures cause the student-teacher divergence to grow super-linearly, making the teacher's token-level supervision unreliable.

SOD adaptively re-weights the distillation strength at each reasoning step based on the divergence trajectory:
- When divergence increases (cascade failure), weights decrease → prevent harmful supervision.
- When the student recovers alignment, weights are restored → maintain effective distillation.

**Paper:** [SOD: Step-wise On-policy Distillation for Small Language Model Agents](https://arxiv.org/abs/2605.07725)

---

## Environment Requirements

| Dependency | Version |
|-----------|---------|
| Python | >= 3.10 |
| CUDA | >= 12.1 (recommended 12.6) |
| PyTorch | >= 2.4 |
| vLLM | >= 0.8.4 (for async rollout) |
| flash-attn | >= 2.5 |
| verl | installed via `pip install -e .` (EasyOPD root) |

### Installation

```bash
# From EasyOPD root directory
pip install -e .
pip install flash-attn --no-build-isolation
pip install vllm
```

Additional dependencies (for reward computation):
```bash
pip install latex2sympy2_extended math_verify
```

---

## File Structure

```
easyopd/methods/sod/
├── __init__.py          # Method registration, metadata, and exports
├── core.py              # Core algorithm implementation
└── README.md            # This file

easyopd/config/
└── sod.yaml             # SOD configuration template (yaml reference)

examples/sod/
└── run_sod.sh           # Complete training launch script
```

### File Descriptions

| File | Description |
|------|-------------|
| `easyopd/methods/sod/core.py` | Core algorithm: `_extract_step_boundaries()` identifies assistant turns from response_mask; `compute_stepwise_opd_weights()` computes per-token w_k weights (Eq. 6, 7); `apply_stepwise_opd()` applies weighted OPD to advantages (Eq. 10). |
| `easyopd/methods/sod/__init__.py` | Exports `compute_stepwise_opd_weights`, `apply_stepwise_opd`, and `SODMethod` class with metadata. |
| `easyopd/config/sod.yaml` | YAML configuration template defining model paths, training hyperparameters, SOD-specific parameters, rollout settings, and data paths. |
| `examples/sod/run_sod.sh` | Bash script that launches the full SOD training pipeline via `python3 -m verl.trainer.main_ppo` with all necessary hydra overrides. |

### Modified verl Files

| File | Modification | Reason |
|------|-------------|--------|
| `verl/trainer/config/algorithm.py` | Added `TokenKLRegConfig` dataclass (enable, coef, gamma, beta_min, beta_max, stepwise_enable, stepwise_epsilon, stepwise_delta, stepwise_opd_coef) | SOD needs config fields for step-wise OPD parameters |
| `verl/trainer/ppo/ray_trainer.py` | Added `_apply_token_kl_regularizer()` method and `_write_stepwise_log()` method; added call site after `compute_advantage()` | Core algorithm entry point that injects weighted OPD into advantages |

---

## Key Equations

- **Eq. 6 (Step Divergence):**
  ```
  d_k = (1/|I_k|) * Σ_{t∈I_k} |log π_θ(y_t) - log π_teacher(y_t)|
  ```
- **Eq. 7 (Adaptive Weight):**
  ```
  w_k = min( ∏_{u=1}^{k-1} (d_u + ε)/(d_{u+1} + ε),  1 + δ )
  ```
- **Eq. 10 (Training Objective):**
  ```
  L = L_GRPO + opd_coef * w_k * (log π_teacher - log π_θ)
  ```

---

## Configuration

### SOD-Specific Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm.token_kl_reg.enable` | bool | `False` | Master switch: enable the token KL regularizer module |
| `algorithm.token_kl_reg.stepwise_enable` | bool | `False` | Enable step-wise mode (SOD core algorithm) |
| `algorithm.token_kl_reg.stepwise_epsilon` | float | `1e-6` | ε in Eq. 7: numerical stability for d_k ratio |
| `algorithm.token_kl_reg.stepwise_delta` | float | `0.5` | δ in Eq. 7: upper bound offset, w_k ≤ 1+δ |
| `algorithm.token_kl_reg.stepwise_opd_coef` | float | `1.0` | Global coefficient for the OPD term in Eq. 10 |
| `algorithm.token_kl_reg.gamma` | float | `1.0` | Legacy: gamma for gated OPD (not used in stepwise mode) |
| `algorithm.token_kl_reg.beta_min` | float | `0.0` | Legacy: minimum beta (not used in stepwise mode) |
| `algorithm.token_kl_reg.beta_max` | float | `None` | Legacy: maximum beta (not used in stepwise mode) |

### Key Training Parameters

| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| `actor_rollout_ref.model.path` | Student SFT ckpt | Student model (HuggingFace format) |
| `+actor_rollout_ref.ref.model.path` | Teacher GRPO ckpt | Teacher model loaded as ref policy |
| `actor_rollout_ref.rollout.multi_turn.enable` | `True` | Required for TIR (tool-integrated reasoning) |
| `actor_rollout_ref.rollout.multi_turn.format` | `hermes` | Chat template format |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | `16` | Max tool interaction rounds |
| `algorithm.adv_estimator` | `grpo` | Advantage estimator (SOD builds on GRPO) |
| `data.max_response_length` | `20480` | Long responses for multi-turn agent tasks |

### How to Pass SOD Parameters via Command Line

```bash
python3 -m verl.trainer.main_ppo \
    +algorithm.token_kl_reg.enable=True \
    +algorithm.token_kl_reg.stepwise_enable=True \
    +algorithm.token_kl_reg.stepwise_epsilon=1e-6 \
    +algorithm.token_kl_reg.stepwise_delta=0.2 \
    +algorithm.token_kl_reg.stepwise_opd_coef=1.0 \
    ... (other parameters)
```

---

## Reproduction Steps

### 1. Data Preparation

Prepare RL training data in verl's expected format (parquet with `chat` column containing multi-turn conversations). The paper uses the Open-AgentRL dataset.

### 2. Model Preparation

- **Student model:** An SFT checkpoint fine-tuned on agent tasks (e.g., Qwen3-0.6B-SFT or Qwen3-1.7B-SFT)
- **Teacher model:** A GRPO-optimized checkpoint with strong reasoning ability (e.g., Qwen3-4B-GRPO)

### 3. Configure and Run

Edit the paths in `examples/sod/run_sod.sh`:
```bash
# Set your model paths
export STUDENT_MODEL_PATH="/path/to/student/checkpoint"
export TEACHER_MODEL_PATH="/path/to/teacher/checkpoint"

# Set your data paths
export OPEN_AGENT_RL="/path/to/training_data.parquet"
export AIME_2024="/path/to/aime2024_eval.parquet"
export AIME_2025="/path/to/aime2025_eval.parquet"

# Launch training
bash examples/sod/run_sod.sh
```

### 4. Hardware Requirements

- **Minimum:** 8× A100 80GB GPUs (1 node)
- **Recommended:** 8× H100 80GB GPUs
- The script uses `infer_tp=4` (4-way tensor parallelism for vLLM rollout) and `train_sp=4` (4-way sequence parallelism for training)

### 5. Monitoring

- WandB logging is enabled by default (set `WANDB_API_KEY`)
- Step-wise OPD weights are logged to `<checkpoint_dir>/stepwise_opd_weights.log`
- Checkpoints saved every 10 steps to `./checkpoint/<experiment_name>/`

---

## Experimental Results (from paper)

| Model | AIME 2024 | AIME 2025 | GPQA | LiveCodeBench |
|-------|-----------|-----------|------|---------------|
| Qwen3-0.6B + SOD | 40.00 | 26.13 | 40.91 | 31.55 |
| Qwen3-1.7B + SOD | 56.67 | 40.00 | 48.99 | 42.86 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'easyopd'` | Run `pip install -e .` from EasyOPD root |
| `KeyError: 'response_mask'` | Ensure `multi_turn.enable=True` in rollout config |
| `KeyError: 'ref_log_prob'` | Ensure `+actor_rollout_ref.ref.model.path` is set (teacher model) |
| OOM during training | Reduce `ppo_mini_batch_size` or enable `param_offload=True` |
| All w_k = 1.0 | Normal for single-step responses; SOD only activates for multi-step |
