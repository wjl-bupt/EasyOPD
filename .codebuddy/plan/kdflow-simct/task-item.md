# 实施计划

- [ ] 1. 梳理并接入 `simct` 方法注册与配置分发
   - 修改 KD 方法选择、配置解析和方法名校验逻辑，确保 `simct` 可被显式选择
   - 为旧名称 `span_ctkd` 增加兼容别名或清晰迁移提示，并在非法方法名时报出可用方法列表
   - 保持未启用 `simct` 时 `simple` 和其它方法默认行为不变
   - _需求：1.1、1.2、1.3、1.4、8.2、8.3_

- [ ] 2. 复用 `simple` teacher 运行链路并抽象 `simct` 差异入口
   - 在现有 teacher sidecar、teacher group、teacher actor 与 SGLang engine 基础上添加 `simct` 所需的轻量参数或策略分支
   - 避免复制大段 `simple` hidden states 生成代码，将 `simct` 专用逻辑封装在独立函数、类或配置分支中
   - 确认 `simple` 路径不新增非预期输出字段或行为差异
   - _需求：2.1、2.2、2.3、2.4、4.2_

- [ ] 3. 实现 `simct` 跨 tokenizer span 对齐核心逻辑
   - 编写 teacher/student token 到文本片段的逐 token 解码与字符边界对齐逻辑
   - 使用 `tokenizer.decode([tid])` 恢复 token 文本，覆盖换行、空格、特殊符号和 Qwen 类 tokenizer 场景
   - 构建 span 级 teacher/student 映射，并对无法可靠对齐的 span 标记为无效或跳过
   - _需求：3.1、3.2、3.3、3.4_

- [ ] 4. 构造 `simct` 后处理输出字段和有效性 mask
   - 在 rollout 后处理阶段生成 `simct` 所需的 teacher hidden states、span 映射、valid mask 与统计字段
   - 明确区分有效 span、无效 span、mask 和 hidden states 对齐结果，保证后续训练步骤可直接消费
   - 处理单 batch 全部 span 无效的情况，避免 silent failure
   - _需求：3.5、4.1、4.4、6.3_

- [ ] 5. 将 `simct` loss 接入训练步骤与指标记录
   - 在现有 KD loss 计算路径中接入 span-level hidden states 蒸馏输入
   - 增加 `simct` 专属 loss、有效 span 数量、有效 span ratio 等指标名称，避免与 `simple` 指标混淆
   - 确保 checkpoint 保存和恢复时 `simct` 配置、状态与训练数据字段保持一致
   - _需求：5.4、5.5、6.4_

- [ ] 6. 增加 `simct` 配置默认值与配置合法性校验
   - 基于 `simple` 配置补充最小差异的 `simct` 配置项和默认值
   - 校验 span 长度、mask 策略、loss 权重和对齐策略等非法配置，并在训练开始前或首次使用前给出清晰错误
   - 确保从 `simple` 切换到 `simct` 只需修改最少必要配置
   - _需求：5.1、5.2、5.3、8.4_

- [ ] 7. 完善 teacher hidden states 请求的错误处理与输入长度处理
   - 在 `simct` teacher 请求失败时附带方法名、批次大小、输入长度、样本索引等关键上下文
   - 复用或补齐 teacher 输入超过上下文长度时的截断、跳过或报错策略，并输出可定位日志
   - 确保训练完成或异常退出时不会因 `simct` 新增资源导致 teacher actor、SGLang engine、W&B 或 DataLoader 清理异常
   - _需求：4.3、4.5、6.2_

- [ ] 8. 添加 `simct` 调试日志和中间结果观测能力
   - 在调试或详细日志模式下输出样本级 span 对齐数量、有效 span 比例和跳过原因统计
   - 为全无效 span、对齐失败、hidden states 缺失等场景增加可定位日志
   - 控制日志开销，避免默认训练路径输出过量信息
   - _需求：6.1、6.2、6.3_

- [ ] 9. 编写 `simct` 单元测试和 `simple` 回归测试
   - 为跨 tokenizer span 对齐、逐 token decode、无效 span 处理和输出字段构造编写单元测试
   - 增加或运行 `simple` 相关最小回归测试，确认新增 `simct` 后现有行为不变
   - 对换行、空格、特殊符号和 Qwen 类 tokenizer 边界样例进行覆盖
   - _需求：7.1、7.2、3.2、3.4_

- [ ] 10. 增加 `simct` smoke test 与示例配置说明
   - 基于 `simple` 示例创建最小差异的 `simct` 示例配置或脚本，说明 `simct` 与原 `span_ctkd` 的对应关系
   - 在无法启动真实 SGLang teacher 的测试环境中使用 mock hidden states 或轻量替身验证 teacher 输出生成与训练数据后处理路径
   - 运行一次短流程 smoke test，确认 `simct` 能完成 teacher 输出生成、后处理和训练数据消费
   - _需求：7.3、7.4、8.1、8.4_
