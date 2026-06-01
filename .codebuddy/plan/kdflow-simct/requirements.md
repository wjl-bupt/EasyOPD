# 需求文档

## 引言

本功能旨在将 KDFlow 中的 `simct` 方法接入当前 EasyOPD/verl 训练流程。`simct` 即原来的 `span_ctkd` 方法，应作为一种与现有 `simple` 方法高度复用的新蒸馏方法加入系统。该功能应尽量沿用 `simple` 方法已有的 teacher sidecar、teacher group、teacher actor、SGLang hidden states 生成、agent loop 后处理与训练配置路径，仅在 `simct` 特有的 span-level / cross-tokenizer KD 逻辑处进行必要扩展。

功能完成后，用户应能够通过配置选择 `simct` 方法，并在不破坏现有 `simple` 方法行为的前提下运行 `simct` 训练、保存检查点、记录指标并完成基本回归验证。

> 约束说明：本需求参考了长期记忆中的跨 Tokenizer 对齐规则。涉及跨 tokenizer 文本对齐时，应使用 `tokenizer.decode([tid])` 进行 token 文本恢复，避免使用 `convert_ids_to_tokens` + `strip` 导致 Qwen 等模型换行或特殊字符对齐断裂。

## 需求

### 需求 1

**用户故事：** 作为一名研究开发者，我希望在 KDFlow/EasyOPD 中通过配置选择 `simct` 方法，以便复现实验中的原 `span_ctkd` 蒸馏流程。

#### 验收标准

1. WHEN 用户在训练配置中指定 KD 方法为 `simct` THEN 系统 SHALL 进入 `simct` 蒸馏流程，而不是回退到 `simple` 或其它方法。
2. WHEN 用户未指定 `simct` 方法 THEN 系统 SHALL 保持现有 `simple` 方法和其它方法的行为不变。
3. WHEN 用户指定不支持的 KD 方法名称 THEN 系统 SHALL 给出清晰的配置错误信息，并列出可用方法。
4. IF 旧配置仍使用 `span_ctkd` 名称 THEN 系统 SHOULD 提供兼容映射或明确提示用户改用 `simct`。

### 需求 2

**用户故事：** 作为一名维护者，我希望 `simct` 最大限度复用 `simple` 的工程结构，以便减少重复代码和降低维护成本。

#### 验收标准

1. WHEN 实现 `simct` 方法 THEN 系统 SHALL 复用 `simple` 中已有的 teacher sidecar、teacher group、teacher actor 与 SGLang engine 的通用能力。
2. WHEN `simct` 与 `simple` 共享 hidden states 生成逻辑 THEN 系统 SHALL 避免复制大段相同实现，并通过参数、策略或轻量扩展承载差异。
3. IF `simct` 需要新增专用处理逻辑 THEN 系统 SHALL 将其封装在清晰的模块、函数或配置分支中，避免污染 `simple` 的默认路径。
4. WHEN `simple` 方法运行现有配置 THEN 系统 SHALL 不因 `simct` 变更产生输出字段、指标、性能或训练行为上的非预期变化。

### 需求 3

**用户故事：** 作为一名算法研究者，我希望 `simct` 能保留原 `span_ctkd` 的 span-level 跨 tokenizer 蒸馏语义，以便学生模型能从 teacher 的 span 表征中学习。

#### 验收标准

1. WHEN `simct` 处理 teacher 与 student 的 token 序列 THEN 系统 SHALL 基于跨 tokenizer 文本对齐结果构建 span 级映射关系。
2. WHEN 跨 tokenizer token 文本需要恢复 THEN 系统 SHALL 使用 `tokenizer.decode([tid])` 逐 token 解码，而不是使用 `convert_ids_to_tokens` 后再 `strip`。
3. IF teacher 与 student 的 span 无法可靠对齐 THEN 系统 SHALL 跳过或标记该样本的无效 span，并避免产生错误的 KD loss 输入。
4. WHEN 样本中包含换行、空格、特殊符号或 Qwen 类 tokenizer 特殊字符 THEN 系统 SHALL 尽可能保持原始文本边界，避免 span 对齐断裂。
5. WHEN 构建 `simct` 训练所需的额外字段 THEN 系统 SHALL 明确区分有效 span、无效 span、mask 和 hidden states 对齐结果。

### 需求 4

**用户故事：** 作为一名训练用户，我希望 `simct` 能沿用当前在线蒸馏训练流程，以便无需重写启动脚本即可进行实验。

#### 验收标准

1. WHEN 用户运行 PPO/agent loop 训练并启用 `simct` THEN 系统 SHALL 在 rollout 后处理阶段生成 `simct` 所需的 teacher 输出。
2. WHEN teacher sidecar 批量请求 hidden states THEN 系统 SHALL 支持 `simct` 所需的批量输入结构，并复用现有 SGLang teacher 推理服务。
3. IF teacher 输入长度超过模型上下文限制 THEN 系统 SHALL 采用现有截断、跳过或报错策略中的一种明确行为，并在日志中提供可定位信息。
4. WHEN `simct` 产生额外 teacher 输出 THEN 系统 SHALL 将其写入后续训练步骤可消费的数据结构中。
5. WHEN 训练完成或异常退出 THEN 系统 SHALL 不因 `simct` 新增资源导致 teacher actor、SGLang engine、W&B 或 DataLoader 清理异常。

### 需求 5

**用户故事：** 作为一名实验用户，我希望 `simct` 提供清晰的配置项和默认值，以便快速从 `simple` 实验迁移到 `simct` 实验。

#### 验收标准

1. WHEN 用户从 `simple` 配置切换到 `simct` THEN 系统 SHALL 只要求修改最少必要配置项。
2. WHEN `simct` 配置未显式指定可选参数 THEN 系统 SHALL 使用合理默认值，并与原 `span_ctkd` 语义保持一致。
3. IF 用户配置了非法的 span 长度、mask 策略、loss 权重或对齐策略 THEN 系统 SHALL 在训练开始前或首次使用前给出清晰错误。
4. WHEN 日志记录训练指标 THEN 系统 SHALL 输出可区分 `simct` 与 `simple` 的 KD 指标名称。
5. WHEN 保存 checkpoint 或恢复训练 THEN 系统 SHALL 保持 `simct` 相关配置与状态可恢复。

### 需求 6

**用户故事：** 作为一名调试者，我希望 `simct` 的中间结果可观测，以便定位 span 对齐、hidden states 生成和 KD loss 的问题。

#### 验收标准

1. WHEN `simct` 运行时开启调试或详细日志 THEN 系统 SHALL 能输出样本级 span 对齐数量、有效 span 比例和跳过原因统计。
2. WHEN teacher hidden states 请求失败 THEN 系统 SHALL 在错误信息中包含方法名、批次大小、输入长度或样本索引等关键上下文。
3. IF 某个 batch 中所有 span 都无效 THEN 系统 SHALL 采用明确策略处理该 batch，并避免 silent failure。
4. WHEN 记录训练指标 THEN 系统 SHALL 包含 `simct` loss、有效 span 数量或有效 span ratio 等关键指标。

### 需求 7

**用户故事：** 作为一名维护者，我希望新增 `simct` 后有最小但有效的测试覆盖，以便确认它没有破坏现有 `simple` 方法。

#### 验收标准

1. WHEN 运行 `simple` 相关现有测试或最小回归脚本 THEN 系统 SHALL 保持原有行为和结果结构不变。
2. WHEN 运行 `simct` 的最小单元测试 THEN 系统 SHALL 覆盖跨 tokenizer span 对齐、无效 span 处理和输出字段构造。
3. WHEN 运行 `simct` 的最小集成测试或 smoke test THEN 系统 SHALL 能完成一次短流程 teacher 输出生成与训练数据后处理。
4. IF 测试环境无法启动真实 SGLang teacher THEN 系统 SHOULD 使用 mock hidden states 或轻量替身验证 `simct` 数据路径。

### 需求 8

**用户故事：** 作为一名项目协作者，我希望 `simct` 的命名、兼容和文档说明清楚，以便理解它与原 `span_ctkd`、现有 `simple` 的关系。

#### 验收标准

1. WHEN 用户查看方法注册、配置示例或文档 THEN 系统 SHALL 明确说明 `simct` 对应原 `span_ctkd` 方法。
2. WHEN 代码中出现方法名判断 THEN 系统 SHALL 优先使用统一的新名称 `simct`。
3. IF 为兼容旧实验保留 `span_ctkd` 别名 THEN 系统 SHALL 在日志或文档中说明该名称已迁移到 `simct`。
4. WHEN 新增示例配置或脚本 THEN 系统 SHALL 基于 `simple` 示例进行最小差异改动，突出需要变更的配置项。
