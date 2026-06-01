# 需求文档：EasyOPD 论文与代码仓库 Main 分支对齐（精选范围）

## 引言

本文档基于对 EasyOPD 论文和代码仓库 main 分支的对比分析，聚焦于当前阶段需要优先完成的改进项。

**本次范围：** 高优先级的 Registry 和 Thin Hooks 架构对齐，以及中优先级的统一配置、Agentic OPD 完善、Diagnostics 系统化、文档更新和系统 benchmark。

**暂不处理：** Black-box/Rubric OPD（ROPD）方法实现、实验数据填充、DSKD/ALM/ULD 等额外方法实现。

### 当前 Main 分支实现现状（最新 pull 后）

已实现的方法模块（`easyopd/methods/`）：
- `simple/` — 基础跨 tokenizer OPD（含 teacher sidecar、alignment、losses）
- `simct/` — SimCT 跨 tokenizer OPD（losses）
- `gkd/` — GKD 标准 logit-based OPD
- `sod/` — SOD Step-wise OPD
- `g_opd/` — G-OPD（Context distillation + corrected reward）
- `opcd/` — OPCD（Context distillation with experience injection）
- `vision_opd/` — Vision-OPD（多模态自蒸馏）
- `sdpo/` — **[新增]** SDPO: Self-Distilled Policy Optimization（自蒸馏，无需外部教师）

当前问题（与上次分析一致，未改善）：
- 所有方法通过 `# [EasyOPD:XXX]` 注释标记的 if 分支直接嵌入 verl 核心文件（`ray_trainer.py` 101.5KB 中有 50+ 处修改）
- 没有 `registry.py` 文件，没有 `from_hparams()` 实现
- 没有统一的 hook 抽象层或配置分发机制
- 每个方法的 `__init__.py` 中有 `XXXMethod` metadata class（声称 "for the EasyOPD registry"），但实际无注册机制
- 新增的 SDPO 方法同样采用了 if 分支嵌入方式（修改了 `core_algos.py`、`actor.py`、`dp_actor.py`、`ray_trainer.py`）

---

## 需求

### 需求 1：统一方法注册表（Registry）

**用户故事：** 作为一名 OPD 研究者，我希望通过统一的注册表机制发现和调用所有已实现的 OPD 方法，以便快速切换和对比不同方法。

#### 验收标准

1. WHEN 用户查看 `easyopd/` 目录 THEN 系统 SHALL 提供一个 `registry.py` 文件，包含方法注册装饰器和方法发现机制
2. WHEN 用户调用 `EasyOPD.from_hparams("method_name", config_path="...")` THEN 系统 SHALL 自动解析方法名并加载对应的方法模块和配置
3. IF 用户指定了一个未注册的方法名 THEN 系统 SHALL 抛出明确的错误信息，列出所有可用方法
4. WHEN 开发者添加新方法 THEN 系统 SHALL 支持通过 `@register_method` 装饰器自动注册，无需修改核心代码
5. WHEN 系统启动 THEN 系统 SHALL 自动扫描 `easyopd/methods/` 下所有已注册方法并构建可用方法列表

**当前差距：** README 中描述了 `registry.py` 和 `from_hparams()` 接口，但代码中实际不存在。各方法的 `__init__.py` 中有 `XXXMethod` metadata class 但没有实际的注册机制。

---

### 需求 2：Thin Hooks 抽象层与代码架构对齐

**用户故事：** 作为一名框架开发者，我希望通过明确定义的 hook 接口将方法逻辑与 verl 执行层解耦，以便新方法可以通过组合 hooks 而非修改 verl 核心代码来接入。

#### 验收标准

1. WHEN 框架初始化 THEN 系统 SHALL 提供以下 hook 接口定义（作为抽象基类或 Protocol）：
   - `LossHook`：token- 或 distribution-level supervision
   - `RolloutHook`：trajectory metadata 附加
   - `RewardHook`：judge 或 rubric feedback
   - `AlignmentHook`：token- 或 span-level mapping
   - `TeacherSidecarHook`：teacher-side signals（logits、hidden states、judge outputs）
2. WHEN 方法需要接入训练流程 THEN 系统 SHALL 通过 hook dispatch 机制路由到方法模块，而非在 verl 代码中硬编码 if 分支
3. IF 一个方法只需要 loss hook THEN 系统 SHALL 允许该方法仅实现 loss hook 而忽略其他 hooks
4. WHEN 新方法被添加 THEN 系统 SHALL 不需要修改 `verl/trainer/ppo/ray_trainer.py` 中的 if 分支逻辑
5. WHEN 查看 verl 核心文件 THEN 系统 SHALL 仅包含少量（≤5 处）通用的 hook dispatch 调用点，而非方法特定的 if 分支
6. WHEN 方法逻辑需要在训练流程中执行 THEN 系统 SHALL 通过 hook dispatch 而非直接代码注入来实现，方法逻辑 100% 在 `easyopd/methods/` 中
7. WHEN 重构完成 THEN 系统 SHALL 将现有 8 个方法（simple、simct、gkd、sod、g_opd、opcd、vision_opd、sdpo）全部迁移到 hook-based 架构，功能不变

**当前差距：** 当前实现采用了"注释标记 + if 分支"的方式将方法逻辑直接嵌入 verl 核心文件。具体分布：
- `ray_trainer.py`（101.5KB）：21 处 `[EasyOPD:XXX]` 标记
- `dp_actor.py`（37KB）：11 处标记
- `actor.py`（15.8KB）：11 处标记
- `algorithm.py`（5.8KB）：7 处标记
- 总计 50 处方法特定的代码注入

---

### 需求 3：统一配置系统与声明式调用

**用户故事：** 作为一名用户，我希望通过修改 yaml 配置文件即可切换 OPD 方法，无需理解底层实现细节，以便快速进行实验。

#### 验收标准

1. WHEN 用户修改 yaml 中的 `method.name` 字段 THEN 系统 SHALL 自动加载对应方法的所有组件（loss、alignment、reward、teacher sidecar）
2. WHEN 用户运行统一的 launch 脚本（如 `run_easyopd.sh`）THEN 系统 SHALL 根据配置自动选择正确的训练流程
3. IF 方法需要特定的超参数 THEN 系统 SHALL 在统一配置空间中暴露这些参数，并提供合理的默认值
4. WHEN 用户对比两个方法 THEN 系统 SHALL 允许仅通过切换 yaml 文件（或命令行 override）实现，而非修改训练脚本
5. WHEN 配置文件被加载 THEN 系统 SHALL 验证所有必需参数已提供，并对缺失参数给出明确错误提示

**当前差距：** `easyopd/config/` 中有 8 个方法的 yaml 文件（simple、simct、gkd、sod、g_opd、opcd、vision_opd、sdpo），但没有统一的配置解析和方法分发机制。每个方法有独立的 `run_xxx.sh` 脚本，切换方法需要使用不同的脚本。

---

### 需求 4：Agentic OPD 完善

**用户故事：** 作为一名 Agent 训练研究者，我希望使用 EasyOPD 进行 trajectory-level 的 agentic OPD 训练，以便在多步推理和工具使用场景中提升学生模型能力。

#### 验收标准

1. WHEN 用户配置 agentic OPD（SOD）THEN 系统 SHALL 支持 multi-step trajectory 的 rollout 和 step-wise supervision
2. WHEN 环境返回 step-level feedback THEN 系统 SHALL 通过 rollout hook 将 trajectory metadata（step boundaries、tool calls、observations）附加到训练 batch
3. WHEN 训练完成 THEN 系统 SHALL 报告 environment-level metrics（task success rate、average steps、invalid action rate、cumulative reward）
4. IF 使用 SOD 方法 THEN 系统 SHALL 支持 tool-integrated reasoning 场景的 step-wise weighting，并通过 sandbox 环境验证工具调用结果
5. WHEN SOD 方法通过 hook 架构接入 THEN 系统 SHALL 保持与当前 SOD 实现相同的训练效果

**当前差距：** SOD 方法已实现基础的 step-wise weighting，但缺少标准化的 agent environment 评测脚本和完整的 trajectory-level metrics 报告。

---

### 需求 5：Method-local Diagnostics 系统化

**用户故事：** 作为一名 OPD 开发者，我希望每个方法都有标准化的诊断指标输出，以便调试训练过程中的异常行为。

#### 验收标准

1. WHEN 方法注册时 THEN 系统 SHALL 要求方法声明其诊断指标列表（通过 `declared_metrics` 属性或类似机制）
2. WHEN 运行 cross-tokenizer OPD THEN 系统 SHALL 报告 overlap vocabulary size、valid aligned positions、alignment coverage
3. WHEN 运行 step-wise OPD THEN 系统 SHALL 报告 step coverage、per-step weight distribution
4. WHEN 运行任何 OPD 方法 THEN 系统 SHALL 自动收集方法声明的 metrics 并通过 wandb/tensorboard 上报
5. IF 方法返回的 metrics 中包含异常值（如 alignment coverage < 10%）THEN 系统 SHALL 输出 warning 日志帮助用户诊断问题

**当前差距：** 部分方法（如 opcd、vision_opd、gkd、sdpo）有 metrics 返回，但没有统一的 diagnostics 框架和标准化的 metrics 注册/收集机制。

---

### 需求 6：文档与论文一致性

**用户故事：** 作为一名新用户，我希望代码仓库的文档与论文描述一致，以便准确理解框架的设计和使用方式。

#### 验收标准

1. WHEN 用户阅读 README THEN 系统 SHALL 展示与论文 Figure 1 一致的三层架构图（User Layer → EasyOPD Layer → verl Layer）
2. WHEN 用户查看方法列表 THEN 系统 SHALL 展示当前支持的所有 OPD regime 及其对应方法
3. WHEN 用户想要快速上手 THEN 系统 SHALL 提供 Quick Start 指南，包含安装、配置、运行的最小步骤
4. IF 论文声称支持某个特性但尚未实现 THEN 文档 SHALL 明确标注为 "Planned" 或 "Coming Soon"
5. WHEN 用户按照文档操作 THEN 系统 SHALL 能够成功运行至少一个完整的 OPD 训练示例

**当前差距：** README 仍然是面向开发者的 git 协作指南（克隆、分支、提交流程），缺少面向用户的框架使用文档、架构图、quick start guide 和方法对比表。

---

### 需求 7：系统效率基准测试

**用户故事：** 作为一名系统研究者，我希望了解 EasyOPD 各方法的系统开销，以便评估框架的实用性。

#### 验收标准

1. WHEN 用户运行 benchmark 脚本 THEN 系统 SHALL 自动测量 training throughput（tokens/sec）
2. WHEN 用户运行 benchmark 脚本 THEN 系统 SHALL 自动测量 peak GPU memory usage
3. WHEN 用户运行 benchmark 脚本 THEN 系统 SHALL 自动测量 wall-clock training time
4. WHEN 对比不同方法 THEN 系统 SHALL 分离 framework overhead 和 method overhead（如 teacher forwarding、alignment construction、reward evaluation 的独立开销）
5. WHEN benchmark 完成 THEN 系统 SHALL 输出结构化的对比报告（表格或 JSON 格式），可直接用于论文 Table 填充

**当前差距：** 没有标准化的 benchmark 脚本来测量和对比不同方法的系统开销。

---

## 总结：本次实施范围

| # | 需求 | 优先级 | 核心工作 |
|---|------|--------|----------|
| 1 | Registry | 🔴 高 | 实现 `registry.py`、`@register_method`、`from_hparams()`，注册全部 8 个方法 |
| 2 | Thin Hooks + 架构对齐 | 🔴 高 | 定义 hook 接口、重构 verl 核心文件（消除 50 处 if 分支）、迁移 8 个方法 |
| 3 | 统一配置 | 🟡 中 | 统一 launch 脚本、配置解析和方法分发 |
| 4 | Agentic OPD 完善 | 🟡 中 | 完善 SOD 评测、trajectory metrics |
| 5 | Diagnostics 系统化 | 🟡 中 | 统一 metrics 注册/收集框架 |
| 6 | 文档更新 | 🟡 中 | 重写 README、添加架构图和 Quick Start |
| 7 | 系统 benchmark | 🟡 中 | 实现 benchmark 脚本和对比报告 |

**暂不处理：**
- ROPD（Black-box OPD）方法实现
- 论文实验数据填充
- DSKD/ALM/ULD 等额外方法实现
