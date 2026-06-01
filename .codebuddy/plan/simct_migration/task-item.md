# 实施计划：simple（cross-tokenizer KD）迁移到 verl

> 基于 `.codebuddy/plan/simct_migration/requirements.md`，按"沿用 verl 现有 OPD 框架 + 最小扩展"策略组织。所有 verl 文件改动用 `# [EasyOPD:simple] ... # [EasyOPD:simple] End` 注释包裹。

- [ ] 1. 搭建 EasyOPD simple 方法骨架与目录结构
   - 创建 `easyopd/methods/simple/` 目录，包含 `__init__.py`、`alignment.py`、`losses.py`、`teacher_forward.py`、`README.md` 五个空文件
   - 在 `__init__.py` 中声明 `SimpleMethod` 元信息（方法名、verl_modified_files 列表、register 入口）
   - 在 `README.md` 中先写好骨架：方法简介、修改了 verl 哪些文件、如何运行
   - _需求：6.1, 6.4, 6.5_

- [ ] 2. 实现 overlap token 发现与序列对齐工具（纯算法，无 verl 依赖）
- [ ] 2.1 在 `easyopd/methods/simple/alignment.py` 实现 `find_overlap_tokens(student_tokenizer, teacher_tokenizer) -> (student_overlap_ids, teacher_overlap_ids)`
   - vocab key 标准化（`Ġ` → `▁`），求交集
   - EOS token 兜底追加
   - 与 KDFlow `simple_ctkd._find_overlap_tokens` 数值等价
   - 编写单元测试：用 Qwen + Llama tokenizer 验证 overlap 数量、EOS 兜底
   - _需求：1.1, 1.2, 1.3, 1.4_

- [ ] 2.2 在同文件实现 `align_sequences(tea_tokens, stu_tokens, tea_eos, stu_eos) -> (tea_aligned_idx, stu_aligned_idx)`
   - 字符级贪心累积对齐（与 KDFlow `_align_sequences` 等价）
   - 同 tokenizer 快速路径（去前缀后相等直接 `range(len)`）
   - EOS 位置对齐特殊处理
   - 编写单元测试：构造 1) 同 tokenizer 样本（恒等对齐）2) 跨 tokenizer 真实样本 3) 完全不能对齐的退化样本
   - _需求：2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 3. 扩展 verl `DistillationLossSettings` 支持第三类 cross-tokenizer loss
   - 修改 `verl/trainer/distillation/losses.py` 中 `DistillationLossSettings` 类：新增 `use_cross_tokenizer: bool = False` 字段
   - 修改 `__post_init__` 互斥校验：`sum([use_topk, use_estimator, use_cross_tokenizer]) != 1` 才报错
   - 所有改动用 `# [EasyOPD:simple] ... # [EasyOPD:simple] End` 注释包裹
   - 验证：既有 `forward_kl_topk` / `kl` 系列注册不报错，行为不变
   - _需求：3.1, 3.2, 3.5, 9.2_

- [ ] 4. 扩展 verl `DistillationConfig` 在 cross-tokenizer 模式下跳过 topk / response_length=1 校验
   - 修改 `verl/workers/config/distillation.py` 中 `DistillationTeacherModelConfig.validate_and_prepare_for_distillation` 签名增加 `use_cross_tokenizer: bool = False` 参数
   - cross-tokenizer 模式下：跳过 `_validate_topk_logprobs` 调用、跳过 `prompt_length += response_length; response_length = 1` 改写
   - 修改 `DistillationConfig.__post_init__` 调用处传入 `use_cross_tokenizer=loss_settings.use_cross_tokenizer`
   - 所有改动用 `# [EasyOPD:simple] ... # [EasyOPD:simple] End` 注释包裹
   - _需求：3.4, 3.5, 4.6, 9.1_

- [ ] 5. 实现 teacher 完整 logits 数据通路（teacher 端重新分词 + HF 前向 + overlap 列裁剪）
   - 在 `easyopd/methods/simple/teacher_forward.py` 实现 `TeacherCrossTokenizerForward` 类
   - 输入：student 的 prompt 字符串 + response 字符串
   - 步骤：teacher tokenizer 重新编码 → HF 模型前向 → 取 response 段 logits → 按 `teacher_overlap_token_ids` 列裁剪
   - 输出字段：`teacher_overlap_logits [B, teacher_resp_len, num_overlap]`、`teacher_input_ids`、`teacher_response_mask`
   - 暴露统一接口 `compute_teacher_overlap_logits(batch) -> dict`，供 verl 训练循环在 rollout 后、loss 前调用
   - 在 `easyopd/methods/simple/README.md` 中明确标注："teacher 走独立 HF 前向，不复用 vllm/sglang 引擎"
   - _需求：4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 6. 实现 simple cross-tokenizer KL loss 函数并注册到 verl
- [ ] 6.1 在 `easyopd/methods/simple/losses.py` 实现 `compute_distillation_loss_simple_cross_tokenizer(config, distillation_config, model_output, data) -> (losses_tensor, metrics)`
   - 取 student logits、`teacher_overlap_logits`、`teacher_input_ids`、`teacher_response_mask`、`response_mask`
   - 逐 sample 调用 `align_sequences`，得到对齐索引
   - 在对齐位置抽取 student overlap 列 logits 与 teacher overlap 列 logits（形状一致断言）
   - 计算 KL（forward 或 reverse，由配置切换）
   - 输出 `[B, resp_len]` losses tensor，未对齐位置填 0；返回 `align_ratio` / `overlap_vocab_size` metrics
   - sample 对齐数为 0 时安全跳过，不产生 NaN
   - **不在内部 reduce**，由上层 `distillation_loss` 统一处理 PG / 监督式两条路径
   - _需求：5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 8.1, 8.2_

- [ ] 6.2 在 `easyopd/methods/simple/losses.py` 暴露 `register_simple_loss()` 函数
   - 函数体内用 `@register_distillation_loss(DistillationLossSettings(names=["simple"], use_cross_tokenizer=True))` 装饰并注册 6.1 的 loss 函数
   - 在 `verl/trainer/distillation/losses.py` 文件**末尾**用 `# [EasyOPD:simple] ... End` 包裹一行 `from easyopd.methods.simple.losses import register_simple_loss; register_simple_loss()`
   - 编写单元测试：注册后 `get_distillation_loss_fn("simple")` 能返回正确函数
   - _需求：5.1, 6.2, 6.3, 6.4_

- [ ] 7. 创建 simple 训练配置文件与运行脚本
   - 新建 `easyopd/config/simple.yaml`：student/teacher 模型路径、`distillation.enabled=true`、`distillation.distillation_loss.loss_mode=simple`、KL 方向（kl/rkl）、`use_policy_gradient` 等参数
   - 新建 `examples/simple/run_simple.sh`：参考 `examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh` 的结构，调用 `python3 -m verl.trainer.main_ppo`
   - 默认配置选用一个跨 tokenizer 对（如 Qwen3-8B → Llama3.2-3B），并提供同 tokenizer 退化场景的注释示例
   - 新建 `examples/simple/README.md`：使用说明、参数解释、预期输出
   - _需求：7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 8. 端到端集成测试与兼容性回归
   - 测试 A（cross-tokenizer 主路径）：用 Qwen3-8B → Llama3.2-3B 小规模数据跑通 `run_simple.sh` 至少 5 step，验证 `align_ratio` > 0、`distillation/loss` 下降趋势合理
   - 测试 B（同 tokenizer 退化）：student 与 teacher 同模型，`loss_mode=simple`，验证 overlap=全 vocab、对齐为恒等、loss 与同 tokenizer KD 数值接近
   - 测试 C（既有 loss 不回归）：分别跑 `loss_mode=k3` 与 `loss_mode=forward_kl_topk` 至少 1 step，验证扩展后行为完全不变
   - 测试 D（关闭 simple）：`distillation.enabled=false` 时跑普通 PPO，确认无 simple 相关代码被加载、无额外开销
   - 把 `align_ratio < 0.5` 的 WARNING 验证一并跑通
   - _需求：7.5, 8.4, 9.1, 9.2, 9.3, 9.4_
