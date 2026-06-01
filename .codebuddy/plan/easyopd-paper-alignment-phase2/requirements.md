# 需求文档：EasyOPD 论文与代码仓库 Main 分支对齐（第二阶段）

## 引言

本文档是 EasyOPD 论文对齐工作的**第二阶段**需求文档。第一阶段已完成了框架核心基础设施的搭建（Registry、Hooks、Config Loader、Diagnostics、文档、Benchmark），本阶段聚焦于完成剩余的深度重构和功能完善工作。

**第一阶段已完成的工作：**
- ✅ `easyopd/registry.py` — 统一方法注册表 + `@register_method` 装饰器 + `auto_discover()`
- ✅ `easyopd/__init__.py` — `EasyOPD` 类 + `from_hparams()` 统一入口
- ✅ `easyopd/hooks.py` — 5 个 Hook Protocol 接口定义（LossHook、RolloutHook、RewardHook、AlignmentHook、TeacherSidecarHook）
- ✅ `easyopd/hook_dispatch.py` — HookDispatcher 路由机制
- ✅ 8 个方法的 `hooks.py` 适配器文件
- ✅ `easyopd/config_loader.py` — 统一配置加载与验证
- ✅ `easyopd/diagnostics.py` — MetricsCollector + 异常检测
- ✅ `scripts/run_easyopd.py` + `scripts/run_easyopd.sh` — 统一 launch 入口
- ✅ `scripts/benchmark_methods.py` — 系统效率 benchmark
- ✅ `README.md` — 面向用户的框架文档（旧版已保存为 `README_DEV.md`）
- ✅ `tests/easyopd/test_registry.py`、`test_hooks.py`、`test_hook_dispatch.py` — 测试套件
- 🔄 verl 核心文件中已添加 HookDispatcher 桥接层初始化代码

**本阶段范围：**
1. 完成 verl 核心文件的 if 分支移除（任务 6 的完整实施）
2. 完善 SOD Agentic OPD 评测脚本（任务 10）

**暂不处理：** ROPD 方法实现、实验数据填充、DSKD/ALM/ULD 等额外方法实现。

---

## 需求

### 需求 1：完成 verl 核心文件 if 分支到 Hook Dispatch 的迁移

**用户故事：** 作为一名框架开发者，我希望 verl 核心文件中的 50 处方法特定 if 分支被替换为通用的 hook dispatch 调用，以便新方法无需修改 verl 代码即可接入。

#### 当前状态

verl 核心文件中已添加 HookDispatcher 初始化代码（桥接层），但 50 处 `[EasyOPD:XXX]` if 分支仍然存在。需要逐步将这些 if 分支的逻辑迁移到对应方法的 hook 实现中，然后用 hook dispatch 调用替换。

涉及的文件和标记数量：
- `verl/trainer/ppo/ray_trainer.py`：21 处标记（G-OPD 初始化、OPCD batch 构建、Vision-OPD batch 构建、SOD 权重计算、G-OPD ref log prob、OPCD experience injection）
- `verl/workers/actor/dp_actor.py`：16 处标记（G-OPD log probs、Vision-OPD self-distillation keys、OPCD context keys、G-OPD advantages、Vision-OPD loss、OPCD KL loss、Vision-OPD EMA update）
- `verl/workers/config/actor.py`：11 处标记（SelfDistillationConfig、G-OPD fields、OPCD config、Vision-OPD config）
- `verl/trainer/config/algorithm.py`：7 处标记（SOD params、G-OPD context distillation、Vision-OPD rollout correction）

#### 验收标准

1. WHEN 重构完成 THEN `verl/trainer/ppo/ray_trainer.py` SHALL 仅包含 ≤5 处通用 hook dispatch 调用点，所有 21 处 `[EasyOPD:XXX]` if 分支 SHALL 被移除
2. WHEN 重构完成 THEN `verl/workers/actor/dp_actor.py` SHALL 仅包含 ≤3 处通用 hook dispatch 调用点，所有 16 处方法特定分支 SHALL 被移除
3. WHEN 重构完成 THEN `verl/workers/config/actor.py` 中的方法特定 config（`SelfDistillationConfig`、G-OPD fields、OPCD fields）SHALL 被移至 `easyopd/methods/` 对应目录，verl 侧仅保留 `easyopd_method_config: Optional[dict]` 通用字段
4. WHEN 重构完成 THEN `verl/trainer/config/algorithm.py` 中的方法特定字段 SHALL 被移至方法自身的 config，verl 侧仅保留 `easyopd: Optional[dict]` 占位
5. WHEN 任何一个方法的 if 分支被移除后 THEN 该方法的训练功能 SHALL 保持不变（通过 hook dispatch 路由实现相同逻辑）
6. WHEN 重构过程中 THEN 系统 SHALL 支持新旧代码共存（渐进式迁移），不要求一次性完成所有方法的迁移
7. IF 某个方法的 hook 适配器尚未完全实现 THEN 系统 SHALL 保留该方法的 if 分支作为 fallback，直到 hook 适配器验证通过

---

### 需求 2：SOD Agentic OPD 评测脚本

**用户故事：** 作为一名 Agent 训练研究者，我希望有标准化的评测脚本来衡量 SOD 方法在 tool-integrated reasoning 场景中的效果。

#### 验收标准

1. WHEN 用户运行 `examples/sod/eval_agent.py` THEN 系统 SHALL 自动执行 agent 在 sandbox 环境中的评测
2. WHEN 评测完成 THEN 系统 SHALL 报告以下 trajectory-level metrics：
   - task success rate（任务完成率）
   - average steps per task（平均步数）
   - invalid action rate（无效动作率）
   - cumulative reward（累积奖励）
3. WHEN 评测脚本运行 THEN 系统 SHALL 支持配置不同的 sandbox 环境（如 code execution、web browsing、tool calling）
4. WHEN 评测结果生成 THEN 系统 SHALL 输出结构化报告（JSON 格式），可直接用于论文 Table 填充
5. IF sandbox 环境不可用 THEN 系统 SHALL 支持 mock 模式进行离线评测（基于预录制的 trajectories）

---

## 总结：本阶段实施范围

| # | 需求 | 优先级 | 核心工作 |
|---|------|--------|----------|
| 1 | verl if 分支迁移 | 🔴 高 | 逐步移除 50 处 if 分支，替换为 hook dispatch 调用 |
| 2 | SOD 评测脚本 | 🟡 中 | 实现 agent 评测脚本和 trajectory metrics |

**迁移策略（需求 1）：** 按方法逐个迁移，每完成一个方法的迁移后运行测试验证功能不变。迁移顺序建议：GKD（最简单，仅 loss）→ SOD（loss + rollout）→ OPCD（loss + rollout）→ G-OPD（loss + reward + teacher）→ Vision-OPD（loss + teacher + EMA）→ SDPO（loss + teacher）→ simple/simct（cross-tokenizer，最复杂）。
