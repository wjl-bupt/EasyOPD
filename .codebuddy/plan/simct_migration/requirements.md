# 需求文档：simple（cross-tokenizer KD）迁移到 verl 框架

## 引言

本需求文档描述将 KDFlow 中最简单的跨 tokenizer 知识蒸馏方法 `simple_ctkd` 迁移到 EasyOPD 项目中 verl 框架的需求。**核心策略：最大化复用 verl 已有的 on-policy distillation 框架，仅在必要处扩展。**

### 命名约定（重要）

EasyOPD 中的 cross-tokenizer KD 方法与 KDFlow 算法的对应关系：

| EasyOPD 方法名 | KDFlow 源算法 | KDFlow 源文件 | 核心思想 |
|----------------|--------------|---------------|---------|
| **simple** | `simple_ctkd` | `kdflow/algorithms/simple_ctkd.py` | 找两个 tokenizer 的重叠 token，字符级贪心对齐 student/teacher 序列，仅在 1:1 对齐位置上、仅在重叠 token 子集上计算 KL loss |
| **simct** | `span_ctkd` | `kdflow/algorithms/span_ctkd.py` | 在 simple 基础上，将无法 1:1 对齐的连续 token 分组成 span，构建"虚拟共同词表"（overlap + spans），在 span 维度上也参与 loss 计算 |

**本次需求范围：仅 `simple` 方法在 verl 上的迁移。** `simct`（span）方法将在 simple 跑通后另行规划，此处不展开。

### 迁移背景

- 源代码：`KDFlow/kdflow/algorithms/simple_ctkd.py`
- 目标框架：verl（EasyOPD 项目中的底层框架）
- 开发分支：`sunjie-simct`（在该分支上同时承载 simple 与后续 simct 的开发）
- 接入模式：模式 C（独立训练框架适配，参考 EasyOPD README 5.2）

### simple 方法核心思想

在 student 与 teacher 使用不同 tokenizer 的场景下：
1. 计算两个 tokenizer 词表的交集 → **overlap tokens**，记录其在各自词表中的 ID 映射；
2. 字符级贪心对齐 teacher / student 的 response token 序列，得到一一对应的位置对（无法对齐的 token 直接丢弃）；
3. 在对齐位置上，从 student/teacher 的完整 logits 中各自抽取 overlap 列，得到形状一致的子集 logits；
4. 在该 overlap 子词表上计算 KL（forward 或 reverse），按 verl 的 `loss_agg_mode` 聚合。

---

## verl 现有 OPD 框架调研结论（接入策略依据）

通过调研 `verl/trainer/distillation/losses.py` 与 `verl/workers/config/distillation.py`，verl 已经构建了完整的 on-policy distillation 框架，本次 simple 迁移的总体策略是 **"沿用框架 + 最小扩展"**。

### 现有架构组件（必须沿用，不重造）

1. **统一注册机制**：`@register_distillation_loss(DistillationLossSettings(...))` 装饰器 + `DISTILLATION_LOSS_REGISTRY` / `DISTILLATION_SETTINGS_REGISTRY`。
2. **统一派发入口**：`distillation_loss(...)` → `get_distillation_loss_fn(loss_config.loss_mode)` → 注册的 loss 函数。
3. **双路径训练策略**：
   - `use_policy_gradient=True`：把 distillation loss 取负作为 advantage，走 `policy_loss_fn`（thinkingmachines blog 风格）；
   - `use_policy_gradient=False`：直接 `agg_loss` 监督式反传（GKD 风格）。
4. **统一 loss 聚合**：`agg_loss(loss_mat, response_mask, loss_agg_mode, **global_batch_info)`。
5. **任务奖励合并**：`use_task_rewards` + `distillation_loss_coef` 控制 distillation 与 PPO 的混合。
6. **Teacher 资源池**：`DistillationTeacherModelConfig` / `teacher_models` dict，独立 GPU 资源池跑 teacher 推理。

### 已注册的两类 loss（simple 是第三类）

| 类别 | 已有注册名 | `use_topk` | `use_estimator` | teacher 数据 | 计算位置 |
|------|-----------|-----------|----------------|-------------|---------|
| Top-K KL | `forward_kl_topk` | ✓ | ✗ | `teacher_logprobs (B,T,topk)`、`teacher_ids (B,T,topk)` | logits processor 内 |
| KL Estimator | `kl/k1/abs/mse/k2/low_var_kl/k3` | ✗ | ✓ | `teacher_logprobs (B,T)`（仅采样 token） | 正常 loss 阶段 |
| **Cross-Tokenizer (simple, 本次新增)** | `simple` | ✗ | ✗ | teacher 在重叠词表列上的完整 logits + teacher 自己的 input_ids/response_mask | 正常 loss 阶段 |

### simple 与现有架构的关键差异点

1. **`DistillationLossSettings` 当前强制 `use_topk` / `use_estimator` 二选一**（`__post_init__` 中 `sum([...]) != 1` 会直接报错）。simple 需要扩展为"三选一"，新增 `use_cross_tokenizer` 字段。
2. **现有 teacher 推理走 vllm/sglang，且强制 `response_length=1`，仅返回 logprobs**。simple 需要 teacher 在 student 完整 response 上做一次前向，并取出重叠词表列的 logits（无法直接复用现有 vllm/sglang 通路，需要新增 teacher 计算路径）。
3. **现有数据通路假设 student 和 teacher 共享 tokenizer**。simple 需要 teacher 用自己的 tokenizer 重新对 student response 文本分词，得到 `teacher_input_ids` / `teacher_response_mask`。

---

## 需求

### 需求 1：跨 Tokenizer 重叠 Token 发现

**用户故事：** 作为一名研究者，我希望系统能自动发现 student 和 teacher tokenizer 之间的重叠 token，以便在跨 tokenizer 蒸馏中确定可计算 loss 的 token 子集。

#### 验收标准

1. WHEN 启用 simple 模式时 THEN 系统 SHALL 在初始化阶段比较 student 和 teacher tokenizer 的词表，找出所有重叠 token，并记录重叠 token 在各自词表中的 ID 映射（`student_overlap_token_ids`、`teacher_overlap_token_ids`）
2. WHEN 两个 tokenizer 的 token 表示格式不同（如 `Ġ` vs `▁`）THEN 系统 SHALL 对 vocab key 进行标准化（统一替换为 `▁`）后再求交集
3. WHEN EOS token 不在重叠集合中 THEN 系统 SHALL 自动将 student 和 teacher 的 EOS token id 追加到各自的 overlap id 列表中
4. WHEN 重叠 token 发现完成后 THEN 系统 SHALL 记录重叠 token 数量到日志（与 KDFlow `Num of overlap_tokens between student & teacher: X` 保持一致语义）
5. WHEN overlap 信息被构建一次后 THEN 系统 SHALL 缓存复用，避免每次 forward 重复计算

---

### 需求 2：跨 Tokenizer 序列对齐

**用户故事：** 作为一名研究者，我希望系统能将 teacher 和 student 的 token 序列在字符级对齐，以便在对应位置上计算蒸馏 loss。

#### 验收标准

1. WHEN 给定 teacher 和 student 的 token 序列时 THEN 系统 SHALL 实现与 KDFlow `simple_ctkd._align_sequences` 等价的字符级贪心对齐算法（基于 `history_tea_seq` / `history_stu_seq` 累积字符串比较）
2. WHEN teacher 与 student 的 token 序列在去除 `▁` / `Ġ` 前缀后完全相同 THEN 系统 SHALL 走快速路径直接返回 `range(len)` 索引
3. WHEN 对齐完成后 THEN 系统 SHALL 返回 `teacher_aligned_idx` 与 `student_aligned_idx` 两个等长列表，使得 `teacher_tokens[teacher_aligned_idx[i]]` 与 `student_tokens[student_aligned_idx[i]]` 在字符级对应
4. WHEN teacher 和 student 序列以各自 EOS 结尾时 THEN 系统 SHALL 将 EOS 位置作为合法对齐对（即使去除前缀后字符串不相等）
5. WHEN 某些 token 无法对齐 THEN 系统 SHALL 直接跳过（不报错），仅保留成功对齐的 token 对

---

### 需求 3：扩展 DistillationLossSettings 支持第三类 loss（最小侵入）

**用户故事：** 作为一名开发者，我希望在 verl 现有 `DistillationLossSettings` 基础上扩展第三类 loss（cross-tokenizer），以便 simple 能够沿用既有注册机制。

#### 验收标准

1. WHEN 修改 `DistillationLossSettings` 时 THEN 系统 SHALL 新增 `use_cross_tokenizer: bool = False` 字段，并将 `__post_init__` 中的互斥校验从"`use_topk`/`use_estimator` 二选一"改为"`use_topk`/`use_estimator`/`use_cross_tokenizer` 三选一"
2. WHEN 既有 loss（`forward_kl_topk` / `kl` 系列）的注册声明保持不变 THEN 系统 SHALL 保证既有 loss 仍正常工作（默认 `use_cross_tokenizer=False`）
3. WHEN `simple` loss 被注册时 THEN 系统 SHALL 使用 `DistillationLossSettings(names=["simple"], use_cross_tokenizer=True)`
4. WHEN 修改 `DistillationConfig.__post_init__` 时 THEN 系统 SHALL 在调用 `validate_and_prepare_for_distillation` 时新增 `use_cross_tokenizer` 参数，使得 cross-tokenizer 模式下跳过 `_validate_topk_logprobs` 与 `response_length=1` 的强制改写
5. WHEN 修改的 verl 文件涉及 `verl/trainer/distillation/losses.py` 与 `verl/workers/config/distillation.py` 时 THEN 系统 SHALL 用 `# [EasyOPD:simple] ... # [EasyOPD:simple] End` 注释包裹所有改动

---

### 需求 4：Teacher 完整 Logits 数据通路

**用户故事：** 作为一名研究者，我希望 teacher 在 student 完整 response 上前向并返回足以覆盖重叠词表的 logits，以便 simple loss 能够拿到所需的 teacher 概率分布。

#### 验收标准

1. WHEN simple 模式启用时 THEN 系统 SHALL **不复用**当前 vllm/sglang 的 `response_length=1` + topk logprobs 路径，而是新增"teacher 在 GPU 上跑一次 HF / Megatron 前向"的数据通路
2. WHEN teacher 接收到 student 的 prompt 与 response 时 THEN 系统 SHALL：
   - 用 student tokenizer 解码 response 文本；
   - 用 teacher 自己的 tokenizer 重新编码，得到 `teacher_input_ids` 与 `teacher_response_mask`；
   - 在 teacher 模型上前向，得到 response 段的 logits（或 hidden_states + lm_head）
3. WHEN teacher 前向完成后 THEN 系统 SHALL 仅保留 `teacher_overlap_token_ids` 对应的列（在 teacher 端做列裁剪）后传出，命名为 `teacher_overlap_logits`，shape 为 `[B, teacher_resp_len, num_overlap]`，以减小通信量
4. WHEN teacher 返回数据时 THEN 系统 SHALL 将以下字段一并放入 batch（与现有 `teacher_logprobs` / `teacher_ids` 字段并列共存，互不冲突）：
   - `teacher_overlap_logits`：[B, teacher_resp_len, num_overlap]
   - `teacher_input_ids`：[B, teacher_resp_len]
   - `teacher_response_mask`：[B, teacher_resp_len]
5. IF 第一阶段实现选择"复用 verl 现有 vllm/sglang 引擎并跳过 response_length=1 改写"路径有困难 THEN 系统 SHALL 退而求其次选择"在 student 训练 worker 内、或新增独立 HF teacher worker 跑前向"路径，并在 simple 的 README 中明确标注该实现选择
6. WHEN 不启用 simple 模式时 THEN 系统 SHALL 保证 teacher 推理路径完全不变（不引入 HF 前向开销，不破坏 vllm/sglang 通路）

---

### 需求 5：simple Cross-Tokenizer KL Loss 注册与计算

**用户故事：** 作为一名研究者，我希望在对齐位置上仅使用重叠词表 logits 计算 KL 散度，并通过 verl 的标准注册机制接入。

#### 验收标准

1. WHEN 注册 simple loss 时 THEN 系统 SHALL 使用 `@register_distillation_loss(DistillationLossSettings(names=["simple"], use_cross_tokenizer=True))` 注册函数 `compute_distillation_loss_simple_cross_tokenizer`
2. WHEN 该 loss 函数被调用时 THEN 系统 SHALL 执行以下步骤：
   - 从 `model_output` 取出 student 完整 logits（或在 logits processor 内提前裁剪到 `student_overlap_token_ids` 列后传入）；
   - 从 `data` 取出 `teacher_overlap_logits`、`teacher_input_ids`、`teacher_response_mask`、student 的 `response_mask`；
   - 对每个 sample 跑序列对齐（需求 2），得到 `(teacher_aligned_idx, student_aligned_idx)`；
   - 在对齐位置上抽取 student 的 overlap 列 logits 与 teacher 的 overlap 列 logits；
   - 计算 KL（forward 或 reverse，由 `loss_config` 子配置切换）
3. WHEN 输出 distillation_losses 时 THEN 系统 SHALL 将其填充到 `[B, resp_len]` 形状的 tensor，未对齐的位置填 0（与 `response_mask` 配合后不会贡献 loss），以保持与既有 loss 函数返回值一致
4. WHEN 单个 sample 的对齐数为 0 THEN 系统 SHALL 安全跳过该 sample，loss 贡献为 0，不导致梯度 NaN
5. WHEN loss 计算完成后 THEN 系统 SHALL 把控制权交还给上层 `distillation_loss(...)`，由它统一调用 `agg_loss` / `policy_loss_fn` 完成最终聚合（**不要在 simple loss 内部做最终 reduce**，保持与 `compute_distillation_loss_reverse_kl_estimator` 等既有函数一致的接口）
6. WHEN 配置中切换 `use_policy_gradient` THEN 系统 SHALL 自动支持 PG / 监督式两条路径（继承自既有 `distillation_loss` 函数，无需 simple 自身处理）

---

### 需求 6：EasyOPD 方法目录结构与对 verl 的最小改动

**用户故事：** 作为一名开发者，我希望 simple 的核心逻辑放在 EasyOPD 自有目录、verl 改动控制在最小范围。

#### 验收标准

1. WHEN 创建 simple 方法目录时 THEN 系统 SHALL 按以下结构组织代码：
   ```
   easyopd/methods/simple/
   ├── __init__.py          # 注册 SimpleMethod 元信息（verl_modified_files 列表）
   ├── alignment.py         # find_overlap_tokens / align_sequences 等纯算法工具
   ├── losses.py            # compute_distillation_loss_simple_cross_tokenizer 实现
   ├── teacher_forward.py   # teacher 端 HF 前向 + 重新分词 + overlap 列裁剪
   └── README.md            # 方法说明（修改了 verl 哪些文件、如何运行）
   ```
2. WHEN 在 verl 中"注册"simple loss 时 THEN 系统 SHALL 在 `verl/trainer/distillation/losses.py` 文件**末尾**添加一段 `# [EasyOPD:simple] ... # [EasyOPD:simple] End` 注释包裹的代码，仅做 `from easyopd.methods.simple.losses import register_simple_loss; register_simple_loss()` 这种触发注册的调用
3. WHEN 修改 verl 配置时 THEN 系统 SHALL 仅修改两处：
   - `verl/trainer/distillation/losses.py` 中 `DistillationLossSettings` 的字段与互斥校验（需求 3.1）；
   - `verl/workers/config/distillation.py` 中 `DistillationConfig.__post_init__` / `DistillationTeacherModelConfig.validate_and_prepare_for_distillation` 跳过 cross-tokenizer 模式下的 topk/response_length=1 校验（需求 3.4）
4. WHEN simple 的核心算法函数（overlap 发现、序列对齐、loss 计算、teacher 前向）被实现时 THEN 系统 SHALL 全部放在 `easyopd/methods/simple/` 下，verl 文件中只做 import 调用，不直接内嵌算法实现
5. WHEN 添加新配置字段时 THEN 系统 SHALL 给出向后兼容的默认值（`use_cross_tokenizer=False`），确保不开启 simple 时 verl 行为完全不变

---

### 需求 7：配置文件与训练脚本

**用户故事：** 作为一名研究者，我希望能通过 yaml 配置和 shell 脚本一键运行 simple 跨 tokenizer 蒸馏实验。

#### 验收标准

1. WHEN 创建 simple 配置文件时 THEN 系统 SHALL 在 `easyopd/config/simple.yaml` 中包含：
   - student 模型 / tokenizer 路径
   - `distillation.enabled=true`、`distillation.distillation_loss.loss_mode=simple`
   - teacher 模型 / tokenizer 路径（teacher 用 HF 前向路径，而非 vllm/sglang）
   - `distillation.distillation_loss.use_policy_gradient` 与 KL 方向（`kl`/`rkl`）等子参数
2. WHEN 创建训练脚本时 THEN 系统 SHALL 在 `examples/simple/run_simple.sh` 提供可直接运行的脚本，并附带 `examples/simple/README.md`
3. WHEN 用户运行训练脚本时 THEN 系统 SHALL 通过 verl 标准入口（如 `main_ppo.py`）启动训练，`loss_mode=simple` 自动派发到 simple loss 函数
4. WHEN 配置中指定不同的 teacher 与 student（如 Qwen3-30B-A3B → Llama3.2-3B）THEN 系统 SHALL 自动启用 cross-tokenizer 路径
5. WHEN student 与 teacher 使用相同 tokenizer 但 `loss_mode=simple` THEN 系统 SHALL 仍可正确运行（overlap = 全 vocab，对齐为恒等映射），结果应与同 tokenizer KD 等价

---

### 需求 8：Metrics 与监控

**用户故事：** 作为一名研究者，我希望训练过程中能监控跨 tokenizer 对齐质量与 loss 数值，以便判断蒸馏效果。

#### 验收标准

1. WHEN 每个训练 step 完成后 THEN 系统 SHALL 在 simple loss 函数返回的 `metrics` 字典中加入：
   - `distillation/align_ratio`：对齐 token 数占 student response token 数的比例
   - `distillation/overlap_vocab_size`：重叠词表大小（常量）
2. WHEN simple loss 返回 metrics 时 THEN 系统 SHALL 沿用 verl 现有 `Metric(AggregationType.X, ...)` 协议，使其能被 wandb / console logger 自动采集
3. WHEN `distillation/loss` 与 `distillation/abs_loss` 等既有 metrics 被沿用时 THEN 系统 SHALL 复用 `distillation_loss` 函数中的通用 metrics 收集逻辑（如 `compute_distillation_loss_range`），不重复造轮子
4. IF `align_ratio` 在若干 step 内持续低于 0.5 THEN 系统 SHALL 在日志中输出 WARNING（提示对齐质量异常）

---

### 需求 9：兼容性与回退

**用户故事：** 作为一名开发者，我希望 simple 的接入不影响 verl 原有功能与其他 EasyOPD 方法。

#### 验收标准

1. WHEN simple 未被启用（`loss_mode != simple`）THEN 系统 SHALL 保持 verl 既有 distillation 行为完全不变（同 tokenizer topk / estimator 路径无回归）
2. WHEN `DistillationLossSettings` 扩展为三选一后 THEN 系统 SHALL 保证 `forward_kl_topk` / `kl` 系列的既有注册声明不变，运行结果与扩展前数值完全一致
3. WHEN simple 相关代码被加载时 THEN 系统 SHALL 不引入额外的必需第三方依赖（仅依赖 PyTorch、HuggingFace tokenizers / transformers 这些已有依赖）
4. WHEN simple 模式下 teacher 走 HF 前向路径时 THEN 系统 SHALL 不影响其他模式仍正常使用 vllm/sglang teacher 推理
5. WHEN 后续 simct（span）方法接入时 THEN 系统 SHALL 与 simple 共存于 `easyopd/methods/` 下，互不依赖、互不影响（simct 复用 simple 的 overlap/对齐工具时通过 import 复用，不允许把 simct 的逻辑塞进 simple 模块）

---

## 范围边界（明确不在本次需求内的事项）

为避免范围蔓延，以下内容**不属于**本次 simple 迁移需求，将在后续单独规划：

1. **simct（span 方法）的迁移**：包括 `_align_sequences_with_spans`、`_build_virtual_vocab_logits`、virtual common vocabulary 构建等 span 特有逻辑，全部留到 simple 跑通后单独立项；
2. **vllm / sglang 引擎对 cross-tokenizer 的原生支持**：本次 simple 用"独立 HF / Megatron teacher 前向 worker"路径实现，引擎层改造留作后续优化；
3. **多模态扩展**：仅处理纯文本场景，不涉及 `mm_*` 参数；
4. **大规模性能优化**：本次以"功能可跑通且与 KDFlow 数值对齐"为目标，性能调优（teacher 列裁剪 fused kernel、CP/TP 切分下的 cross-tokenizer 通信优化等）属于后续工作。
