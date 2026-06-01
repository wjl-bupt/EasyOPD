# 实施计划：EasyOPD 第二阶段 — verl if 分支迁移 + SOD 评测

## 依赖关系

```
任务1 (GKD迁移) ──→ 任务2 (SOD迁移) ──→ 任务3 (OPCD迁移) ──→ 任务4 (G-OPD迁移)
                                                              ──→ 任务5 (Vision-OPD迁移)
                                                              ──→ 任务6 (SDPO迁移)
                                                              ──→ 任务7 (simple/simct迁移)
任务8 (config迁移) 可与任务1-7并行
任务9 (SOD评测) 依赖任务2完成
任务10 (清理验证) 依赖任务1-8全部完成
```

---

- [ ] 1. 迁移 GKD 方法：移除 verl 中的 GKD if 分支
   - 在 `verl/trainer/distillation/losses.py` 中，将 GKD loss mode 的注册逻辑迁移到 `easyopd/methods/gkd/hooks.py` 的 `GKDLossHook.compute_loss()` 中
   - 在 `verl/workers/config/distillation.py` 中，将 `gkd_beta` 字段移至 `easyopd/methods/gkd/` 的方法配置中
   - 在 `dp_actor.py` 中用 `hook_dispatcher.compute_loss()` 替换 GKD loss 分支
   - 编写验证测试：确保 GKD 通过 hook dispatch 产生与原 if 分支相同的 loss 输出
   - _需求：1.1, 1.2, 1.5_

- [ ] 2. 迁移 SOD 方法：移除 verl 中的 SOD if 分支
   - 在 `ray_trainer.py` 中移除 `[EasyOPD:SOD] Step-wise OPD weighting` 代码块（行 1218-1252），替换为 `hook_dispatcher.compute_loss()` 调用
   - 在 `verl/trainer/config/algorithm.py` 中将 `stepwise_*` 字段（行 70 附近的 TokenKLRegConfig）移至 `easyopd/methods/sod/` 的方法配置
   - 更新 `SODLossHook` 和 `SODRolloutHook` 以完整实现原 if 分支中的逻辑
   - 编写验证测试：确保 SOD step-wise weighting 通过 hook dispatch 产生相同结果
   - _需求：1.1, 1.4, 1.5_

- [ ] 3. 迁移 OPCD 方法：移除 verl 中的 OPCD if 分支
   - 在 `ray_trainer.py` 中移除 `[EasyOPD:OPCD]` 代码块（行 645-826 batch 构建 + 行 2031-2043 experience injection），替换为 `hook_dispatcher.on_rollout_end()` 调用
   - 在 `dp_actor.py` 中移除 `[EasyOPD:OPCD]` 代码块（行 425-431 context keys + 行 624-672 KL loss），替换为 hook dispatch 调用
   - 在 `verl/workers/config/actor.py` 中将 OPCD config 字段（行 195-199）移至 `easyopd/methods/opcd/` 的方法配置
   - 更新 `OPCDLossHook` 和 `OPCDRolloutHook` 以完整实现原 if 分支中的逻辑
   - _需求：1.1, 1.2, 1.3, 1.5_

- [ ] 4. 迁移 G-OPD 方法：移除 verl 中的 G-OPD if 分支
   - 在 `ray_trainer.py` 中移除 `[EasyOPD:G-OPD]` 代码块（行 428-464 初始化 + 行 1893-1960 ref log prob + context distillation），替换为 hook dispatch 调用
   - 在 `dp_actor.py` 中移除 `[EasyOPD:G-OPD]` 代码块（行 388-403 log probs + 行 485-527 advantages），替换为 `hook_dispatcher.compute_loss()` 调用
   - 在 `verl/workers/config/actor.py` 中将 G-OPD fields（行 129-133 `only_reverse_kl_advantages`、`lambda_vals`、`multi_teacher_distill`）移至 `easyopd/methods/g_opd/` 的方法配置
   - 在 `verl/trainer/config/algorithm.py` 中将 G-OPD context distillation 字段（行 107-114）移至方法配置
   - 更新 `GOPDLossHook`、`GOPDRewardHook`、`GOPDTeacherSidecarHook` 以完整实现原逻辑
   - _需求：1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 5. 迁移 Vision-OPD 方法：移除 verl 中的 Vision-OPD if 分支
   - 在 `ray_trainer.py` 中移除 `[EasyOPD:Vision-OPD]` 代码块（行 828-1099 batch 构建 + 行 2021-2029），替换为 hook dispatch 调用
   - 在 `dp_actor.py` 中移除 `[EasyOPD:Vision-OPD]` 代码块（行 405-423 keys + 行 533-622 loss + 行 726-742 EMA update），替换为 hook dispatch 调用
   - 在 `verl/workers/config/actor.py` 中将 `SelfDistillationConfig`（行 31-102）和 Vision-OPD config（行 210-212）移至 `easyopd/methods/vision_opd/` 的方法配置
   - 在 `verl/trainer/config/algorithm.py` 中将 `RolloutCorrectionConfig`（行 115-144）移至方法配置
   - 更新 `VisionOPDLossHook` 和 `VisionOPDTeacherSidecarHook` 以完整实现原逻辑（含 EMA update）
   - _需求：1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 6. 迁移 SDPO 方法：移除 verl 中的 SDPO if 分支
   - 在 `verl/trainer/ppo/core_algos.py` 中移除 SDPO 注册的 `compute_self_distillation_loss` 函数，迁移到 `easyopd/methods/sdpo/hooks.py`
   - 在 `dp_actor.py` 中移除 SDPO self-distillation forward pass 分支，替换为 hook dispatch 调用
   - 在 `ray_trainer.py` 中移除 `_maybe_build_self_distillation_batch` 中 SDPO 相关逻辑，替换为 `hook_dispatcher.teacher_forward()` 调用
   - 在 `verl/workers/config/actor.py` 中将 SDPO 相关的 `SelfDistillationConfig` 字段移至 `easyopd/methods/sdpo/` 的方法配置
   - 更新 `SDPOLossHook` 和 `SDPOTeacherSidecarHook` 以完整实现原逻辑
   - _需求：1.1, 1.2, 1.3, 1.5_

- [ ] 7. 迁移 simple/simct 方法：移除 verl 中的 cross-tokenizer if 分支
   - 在 `verl/trainer/distillation/losses.py` 中移除 `[EasyOPD:simple]` 标记的 `use_cross_tokenizer` 字段和 tail import
   - 在 `verl/workers/config/distillation.py` 中移除 `[EasyOPD:simple]` 标记的 cross-tokenizer mode 特殊处理
   - 将 cross-tokenizer loss 注册逻辑完全移入 `easyopd/methods/simple/hooks.py` 和 `easyopd/methods/simct/hooks.py`
   - 确保 `SimpleLossHook`、`SimpleAlignmentHook`、`SimpleTeacherSidecarHook` 完整实现原有的 logit-processor 两阶段数据流
   - _需求：1.1, 1.2, 1.5_

- [ ] 8. 统一 verl config 文件：移除方法特定字段
   - 在 `verl/workers/config/actor.py` 中用单一 `easyopd_method_config: Optional[dict] = None` 字段替换所有方法特定字段（SelfDistillationConfig、G-OPD fields、OPCD fields、Vision-OPD fields）
   - 在 `verl/trainer/config/algorithm.py` 中用单一 `easyopd: Optional[dict] = None` 字段替换所有方法特定字段（TokenKLRegConfig、G-OPD context distillation、RolloutCorrectionConfig）
   - 更新 `HookDispatcher.from_config()` 以从新的通用字段中提取方法配置
   - 确保所有方法的 yaml 配置文件仍然能正确加载（通过 config_loader 映射到新字段）
   - _需求：1.3, 1.4_

- [ ] 9. 实现 SOD Agentic OPD 评测脚本
   - 创建 `examples/sod/eval_agent.py`：实现标准化的 agent 评测入口，支持 `--env` 参数选择 sandbox 环境类型
   - 实现 trajectory-level metrics 计算：task success rate、average steps、invalid action rate、cumulative reward
   - 实现 mock 模式：支持从 JSON 文件加载预录制 trajectories 进行离线评测
   - 实现结构化报告输出：JSON 格式 + 可选 Markdown table，包含 per-task 和 aggregate metrics
   - 编写 `examples/sod/README.md`：评测脚本使用说明和参数文档
   - _需求：2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 10. 最终清理与集成验证
   - 运行完整测试套件（`tests/easyopd/test_registry.py`、`test_hooks.py`、`test_hook_dispatch.py`），确保所有测试通过
   - 验证 `ray_trainer.py` 中 `[EasyOPD:XXX]` 标记数量 ≤5（仅保留通用 hook dispatch 调用点）
   - 验证 `dp_actor.py` 中 `[EasyOPD:XXX]` 标记数量 ≤3
   - 更新任务清单状态，标记所有已完成任务
   - _需求：1.1, 1.2, 1.5, 1.6, 1.7_
