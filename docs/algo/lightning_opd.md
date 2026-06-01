# Lightning-OPD: Offline On-Policy Distillation

## 方法定义

Lightning-OPD 是 NVIDIA Jet AI 团队提出的离线 on-policy distillation 方法（arXiv:2604.13010）。核心思想是**预计算 teacher log-probabilities**，从而在训练期间完全消除对 live teacher 服务的依赖，将标准 OPD 的训练成本降低 3.6–4.0×。

### 与标准 OPD 的差异

| 特性 | 标准 OPD | Lightning-OPD |
|---|---|---|
| Teacher 推理 | 训练时实时调用 teacher | 离线预计算，训练时从 parquet 读取 |
| GPU 分配 | 需要 teacher GPU + rollout GPU | 所有 GPU 给 actor |
| MoE 支持 | 30B-A3B 可能 OOM | 可训练 |
| 训练速度 | 基线 | 3.6–4.0× 加速 |

## 先决条件

> **Teacher Consistency 约束**：SFT teacher 和 OPD teacher **必须是同一个模型**。使用不同 teacher 会产生不可消除的 gradient bias（论文 §3）。

在使用 Lightning-OPD 前，确保：
1. SFT 阶段使用的 teacher 模型与 OPD 阶段的 teacher 模型相同。
2. `prepare_data.sh` 会自动检查 teacher consistency；不一致时拒绝继续。
3. 可通过 `--allow-teacher-mismatch` 显式覆盖（仅用于调试）。

## 数据流水线（Step 0–6）

| Step | 描述 | EasyOPD 入口 | 关键环境变量 |
|---|---|---|---|
| 0 | 准备 SFT prompts | `examples/lightning_opd_trainer/tools/prepare_sft_prompts.sh` | `LIGHTNING_OPD_HF_DATASET`, `LIGHTNING_OPD_OUT` |
| 1 | 生成 SFT 训练数据 | `examples/lightning_opd_trainer/tools/generate_sft_data.sh` | `LIGHTNING_OPD_TEACHER_MODEL`, `LIGHTNING_OPD_TEACHER_CHAT_URL`, `LIGHTNING_OPD_SFT_PROMPTS` |
| 2 | SFT 训练 | `examples/lightning_opd_trainer/tools/run_sft.sh` | `LIGHTNING_OPD_SFT_BASE_MODEL`, `LIGHTNING_OPD_SFT_DATA` |
| 3 | 学生 rollout | `examples/lightning_opd_trainer/tools/collect_rollouts.sh` | `LIGHTNING_OPD_SFT_CHECKPOINT`, `LIGHTNING_OPD_STUDENT_URL`, `LIGHTNING_OPD_OPD_PROMPTS` |
| 4 | Teacher logprob 预计算 | `examples/lightning_opd_trainer/tools/prepare_data.sh` | `LIGHTNING_OPD_TOKENIZER`, `LIGHTNING_OPD_ROLLOUTS`, `LIGHTNING_OPD_TEACHER_URL` |
| 5 | Lightning-OPD 训练 | `examples/lightning_opd_trainer/train_lightning_opd.sh` | `LIGHTNING_OPD_SFT_CHECKPOINT`, `LIGHTNING_OPD_DATA`, `MODEL_SCALE` |
| 6 | Megatron→HF 转换 | `examples/lightning_opd_trainer/tools/convert_megatron_to_hf.sh` | `MEGATRON_CKPT_DIR`, `HF_OUTPUT_DIR` |

## Parquet Schema 约定

Step 4 输出的 Lightning-OPD 训练数据 parquet 列定义：

| 列名 | 类型 | 描述 |
|---|---|---|
| `prompt` | str | `apply_chat_template` 输出 |
| `label` | str | 可选；下游评估用 |
| `response_tokens` | list[int] | 学生 rollout 的 response token ids |
| `response_length` | int | response token 数（截断到 `max_response_len`） |
| `teacher_log_probs` | list[float] | 长度 == `response_length`；按 response token 顺序排列 |
| `metadata` | dict | 可选附加信息（`sft_teacher_id`、`opd_teacher_id`） |

## Advantage Estimator

Lightning-OPD 使用 `on_policy_distillation` advantage estimator：

```
advantage = teacher_log_prob - student_log_prob
```

使用方式：在训练命令中设置 `algorithm.adv_estimator=on_policy_distillation`。

Batch 列要求：
- `batch.batch["old_log_probs"]`：student log probabilities（自动计算）
- `batch.batch["teacher_log_probs"]`：precomputed teacher log probabilities（从 parquet 读取）
- `batch.batch["response_mask"]`：response token mask

## 配置与环境变量

### 环境变量（`LIGHTNING_OPD_*` 前缀）

| 变量 | 描述 | 默认值 |
|---|---|---|
| `LIGHTNING_OPD_PROJECT_ROOT` | 项目根目录覆盖 | 自动检测 |
| `LIGHTNING_OPD_DRYRUN` | dry-run 模式 | `false` |
| `LIGHTNING_OPD_SFT_CHECKPOINT` | SFT checkpoint 路径 | 必填 |
| `LIGHTNING_OPD_DATA` | 预计算 parquet 路径 | 必填 |
| `LIGHTNING_OPD_TEACHER_MODEL` | Teacher 模型路径 | 可选 |
| `LIGHTNING_OPD_TEACHER_URL` | Teacher logprob completions URL（Step 4） | `http://127.0.0.1:8000/v1/completions` |
| `LIGHTNING_OPD_TEACHER_CHAT_URL` | Teacher generation chat URL（Step 1） | `http://127.0.0.1:8000/v1/chat/completions` |
| `LIGHTNING_OPD_STUDENT_URL` | SFT student OpenAI/vLLM 服务 URL | `http://127.0.0.1:8000/v1/chat/completions` |
| `MODEL_SCALE` | 模型规模 | `4b` |

### 配置文件

- `easyopd/config/lightning_opd/base.yaml`：训练顶层默认
- `easyopd/config/lightning_opd/training.yaml`：训练超参模板
- `easyopd/config/lightning_opd/data_prep.yaml`：数据准备默认参数
- `easyopd/config/lightning_opd/sft.yaml`：SFT warmup 配方（论文 §3.2）

## 与论文结果的差异说明

- 默认使用 vLLM backend，与论文中使用的 sglang 在数值精度上可能有微小差异。
- 30B-A3B 配置在第一阶段未端到端验证；以 4B/8B 为参考配置。
- 如需精确复现论文数字，参考源仓库的 sglang 路径。

## 与标准 OPD 的关系

EasyOPD 已有 `examples/on_policy_distillation_trainer/` 覆盖标准 OPD（live teacher）。Lightning-OPD 与之并存：

- **标准 OPD**：训练时实时调用 teacher 模型，需要额外 teacher GPU。
- **Lightning-OPD**：离线预计算 teacher logprobs，训练时无需 teacher GPU。

用户可根据硬件条件和训练规模选择合适的方式。

## 引用

```bibtex
@article{lightning_opd2026,
  title={Lightning-OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation},
  author={NVIDIA Jet AI},
  journal={arXiv preprint arXiv:2604.13010},
  year={2026}
}
```
