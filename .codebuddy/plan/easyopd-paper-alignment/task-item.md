# 实施计划：EasyOPD 论文与代码仓库 Main 分支对齐

## 依赖关系

```
需求1 (Registry) ──→ 需求2 (Hooks) ──→ 需求3 (统一配置)
                                    ──→ 需求5 (Diagnostics)
                                    ──→ 需求4 (Agentic OPD)
需求6 (文档) 可并行
需求7 (Benchmark) 依赖需求3完成后
```

---

- [x] 1. 实现方法注册表核心模块 `easyopd/registry.py`
   - 创建 `easyopd/registry.py`，实现 `_REGISTRY` 全局字典、`register_method(name)` 装饰器、`get_method(name)` 查找函数、`list_methods()` 列表函数
   - 实现 `auto_discover()` 函数：扫描 `easyopd/methods/` 下所有子目录，自动 import 其 `__init__.py` 触发注册
   - 对未注册方法名抛出 `MethodNotFoundError`，错误信息中列出所有可用方法
   - _需求：1.1, 1.3, 1.4, 1.5_

- [x] 2. 为现有 8 个方法添加 `@register_method` 装饰器
   - 修改 `easyopd/methods/simple/__init__.py`，为 `SimpleMethod` 类添加 `@register_method("simple")` 装饰器
   - 修改 `easyopd/methods/simct/__init__.py`，为 `SimCTMethod` 类添加装饰器（需先创建 metadata class）
   - 修改 `easyopd/methods/gkd/__init__.py`，为 `GKDMethod` 添加装饰器
   - 修改 `easyopd/methods/sod/__init__.py`，为 `SODMethod` 添加装饰器
   - 修改 `easyopd/methods/g_opd/__init__.py`，为 `GOPDMethod` 添加装饰器
   - 修改 `easyopd/methods/opcd/__init__.py`，为 `OPCDMethod` 添加装饰器
   - 修改 `easyopd/methods/vision_opd/__init__.py`，为 `VisionOPDMethod` 添加装饰器
   - 修改 `easyopd/methods/sdpo/__init__.py`，为 `SDPOMethod` 添加装饰器
   - _需求：1.4, 1.5_

- [x] 3. 实现 `EasyOPD.from_hparams()` 统一入口
   - 在 `easyopd/__init__.py` 中实现 `EasyOPD` 类，提供 `from_hparams(method_name, config_path=None, **overrides)` 类方法
   - `from_hparams` 内部调用 `registry.get_method(name)` 获取方法类，加载对应 yaml 配置，实例化并返回 trainer wrapper
   - 编写单元测试 `tests/easyopd/test_registry.py`：验证注册、发现、from_hparams 调用、未知方法报错
   - _需求：1.2, 1.3_

- [x] 4. 定义 Hook 抽象接口（Protocol 类）
   - 创建 `easyopd/hooks.py`，使用 `typing.Protocol` 定义 5 个 hook 接口：
     - `LossHook`：`compute_loss(student_logits, teacher_logits, mask, config) -> (loss, metrics)`
     - `RolloutHook`：`on_rollout_end(batch, config) -> batch`（附加 trajectory metadata）
     - `RewardHook`：`compute_reward(batch, config) -> rewards`
     - `AlignmentHook`：`build_alignment(student_tokenizer, teacher_tokenizer, input_ids) -> alignment_map`
     - `TeacherSidecarHook`：`teacher_forward(batch, teacher_model, config) -> teacher_outputs`
   - 创建 `easyopd/hook_dispatch.py`，实现 `HookDispatcher` 类：根据注册的方法实例，路由到对应 hook 实现
   - _需求：2.1, 2.2, 2.3_

- [x] 5. 为现有方法实现 Hook 接口适配
   - 5.1 为 `simple` 方法实现 `LossHook`（包装 `losses.py`）、`AlignmentHook`（包装 `alignment.py`）、`TeacherSidecarHook`（包装 `teacher_sidecar.py`）
   - 5.2 为 `simct` 方法实现 `LossHook`（包装 `losses.py` 中的 cross-tokenizer loss）
   - 5.3 为 `gkd` 方法实现 `LossHook`（包装 `core.py` 中的 generalized JSD）
   - 5.4 为 `sod` 方法实现 `LossHook` + `RolloutHook`（包装 `core.py` 中的 step-wise weighting）
   - 5.5 为 `g_opd` 方法实现 `LossHook` + `RewardHook` + `TeacherSidecarHook`（包装 context distillation + corrected reward）
   - 5.6 为 `opcd` 方法实现 `LossHook` + `RolloutHook`（包装 context distillation with experience injection）
   - 5.7 为 `vision_opd` 方法实现 `LossHook` + `TeacherSidecarHook`（包装多模态自蒸馏）
   - 5.8 为 `sdpo` 方法实现 `LossHook` + `TeacherSidecarHook`（包装 self-distillation loss + EMA teacher）
   - _需求：2.6, 2.7_

- [~] 6. 重构 verl 核心文件：用 hook dispatch 替换 if 分支（已添加桥接层，if 分支移除待后续迭代）
   - 6.1 在 `verl/trainer/ppo/ray_trainer.py` 中添加 ≤5 处通用 hook dispatch 调用点（`pre_training_step`、`post_rollout`、`compute_loss`、`post_training_step`、`build_teacher_batch`），移除 21 处 `[EasyOPD:XXX]` if 分支
   - 6.2 在 `verl/workers/actor/dp_actor.py` 中用 `LossHook` dispatch 替换 11 处方法特定的 loss 计算分支
   - 6.3 在 `verl/workers/config/actor.py` 中将方法特定的 config dataclass（如 `SelfDistillationConfig`）移至 `easyopd/methods/` 对应目录，verl 侧仅保留通用的 `easyopd_method_config: Optional[dict]` 字段
   - 6.4 在 `verl/trainer/config/algorithm.py` 中将方法特定字段（G-OPD critique、Vision-OPD rollout_correction）移至方法自身的 config，verl 侧仅保留 `easyopd: Optional[dict]` 占位
   - _需求：2.4, 2.5, 2.6_

- [x] 7. 编写 hook 架构集成测试
   - 创建 `tests/easyopd/test_hooks.py`：验证每个方法的 hook 实现能正确被 dispatch 调用
   - 创建 `tests/easyopd/test_hook_dispatch.py`：验证 HookDispatcher 路由逻辑、多 hook 组合、缺失 hook 的 graceful fallback
   - 扩展 `tests/easyopd/test_integration.py`：验证重构后 8 个方法的端到端训练流程功能不变（使用 mock model）
   - _需求：2.7_

- [x] 8. 实现统一配置解析与 launch 脚本
   - 创建 `easyopd/config_loader.py`：实现 `load_method_config(yaml_path)` 函数，解析 yaml 中的 `method.name` 字段，自动合并方法默认配置与用户 override
   - 实现配置验证：检查必需参数、类型校验、未知参数 warning
   - 创建 `scripts/run_easyopd.sh`：统一 launch 脚本，接受 `--method` 和 `--config` 参数，自动路由到正确的训练入口
   - 创建 `scripts/run_easyopd.py`：Python 入口点，调用 `EasyOPD.from_hparams()` 并启动训练
   - _需求：3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 9. 实现 Diagnostics 统一框架
   - 创建 `easyopd/diagnostics.py`：实现 `MetricsCollector` 类，支持方法通过 `declared_metrics` 属性声明指标
   - 在 `MetricsCollector` 中实现异常值检测（configurable thresholds）和 warning 日志输出
   - 在 hook dispatch 的 `post_training_step` 中自动收集各方法返回的 metrics 并上报 wandb/tensorboard
   - 为每个方法的 metadata class 添加 `declared_metrics` 属性（列出该方法的诊断指标名称和描述）
   - _需求：5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 10. 完善 SOD Agentic OPD 评测（待后续迭代）
   - 在 `examples/sod/` 中添加 `eval_agent.py`：标准化的 agent environment 评测脚本，支持 sandbox 环境调用
   - 实现 trajectory-level metrics 报告：task success rate、average steps、invalid action rate、cumulative reward
   - 确保 SOD 的 `RolloutHook` 实现正确附加 step boundaries 和 tool call metadata
   - _需求：4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 11. 重写 README 与文档更新
   - 重写 `README.md`：面向用户的框架介绍，包含三层架构图（ASCII art 或 mermaid）、支持的 OPD regime 列表、Quick Start 指南
   - 将当前的开发者协作指南移至 `CONTRIBUTING.md`（已存在，合并内容）
   - 在 README 中添加方法对比表（method name、regime、teacher type、key feature、status）
   - 对尚未实现的特性（如 ROPD、DSKD）标注 "Planned"
   - _需求：6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 12. 实现系统效率 benchmark 脚本
   - 创建 `scripts/benchmark_methods.py`：自动运行各方法的短训练（如 10 steps），收集 throughput、peak memory、wall-clock time
   - 实现 overhead 分离：通过 profiling hooks 分别测量 teacher forwarding、alignment construction、loss computation 的耗时
   - 输出结构化对比报告（JSON + Markdown table），可直接用于论文 Table 填充
   - _需求：7.1, 7.2, 7.3, 7.4, 7.5_
