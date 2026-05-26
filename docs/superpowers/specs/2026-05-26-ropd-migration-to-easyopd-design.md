# ROPD 迁移到 EasyOPD 的总体设计

## 背景

当前有两个相关仓库：

- 目标仓库：`/mnt/d/Area/DL/projects/research/EasyOPD`
- 源仓库：`/mnt/d/Area/DL/projects/research/black-opd`

`EasyOPD` 是基于 `verl` 的方法集合仓库，已有 `easyopd/methods/simple`、`easyopd/methods/simct`、`easyopd/methods/sod` 等方法级目录，也已有 `examples/on_policy_distillation_trainer` 和 `tests/easyopd` 这样的入口与测试布局。

`black-opd` 是 ROPD 研究原型仓库，包含算法实现、judge/provider 基础设施、rubricator/verifier prompts、teacher index 与数据构建工具、训练脚本、验证脚本、实验评估脚本和历史文档。该仓库中过去出现过 `shared-rubrics` / `shared_rubrics` 命名；迁移到 EasyOPD 时，该语义统一映射为 `ropd`，不在目标侧继续保留为独立命名。该仓库过去已经做过 surface cleanup：`pyproject.toml` / `uv.lock` 是依赖主线，旧的 `requirements*`、`setup.py`、Docker 入口和 Git LFS 依赖不是迁移依据。

本次目标不是把 `black-opd` 原样嵌入 `EasyOPD`，而是把 ROPD 收敛成 EasyOPD 中一个正式、可维护、路径清晰的方法模块。

## 目标

1. 将 ROPD 作为 EasyOPD 的一等方法迁移到 `easyopd/methods/ropd/`。
2. 将所有 ROPD 可执行入口收敛到同一个执行目录，避免散落在仓库根目录。
3. 将配置、prompt、工具脚本、测试和文档放到 EasyOPD 现有结构能够解释的位置。
4. 保留 ROPD 当前研究功能的行为边界，包括 rubric-based reward、judge provider resolution、dry-run 训练入口和核心训练配置语义。
5. 将旧 `black_opd`、`shared-rubrics`、`shared_rubrics` 命名降级为源仓库历史语义；目标侧新路径、新配置、新文档和主要代码逻辑统一直接使用 `ropd`。
6. 不引入运行产物、缓存、大数据集快照或旧仓库的临时实验状态。
7. 用 CPU 级合同测试和 dry-run 入口测试证明迁移后的结构可导入、可配置、可执行到命令生成阶段。

## 非目标

第一阶段迁移不做以下事情：

- 不重写 ROPD 算法语义。
- 不重新设计 rubricator/verifier prompt schema。
- 不迁移 `outputs/`、`logs/`、`wandb/`、`.venv/`、`.cache/`、`.pytest_cache/`、`__pycache__/`。
- 不迁移源仓库中的完整数据集目录作为默认 repo 内容。
- 不把 `black-opd` 的顶层 `algo/`、`training/`、`prompts/`、`data/`、`experiments/` 原样复制到 EasyOPD 根目录。
- 不覆盖 EasyOPD 的 `pyproject.toml`、`requirements*.txt` 或 `setup.py`。
- 不在第一阶段删除 EasyOPD 现有 OPD / SOD / SimCT 功能。
- 不迁移、保留或整理 ROPD 主线之外的其他 ablation；这些内容留在源仓库作为历史参考。
- 不准备直接向 upstream `verl-project/verl` 发 PR；如果未来要发 PR，需要按 AGENTS.md 做 duplicate-work checks 和人工责任说明。

## 总体迁移原则

### 1. 方法归方法，入口归入口

ROPD 的 Python 主实现进入方法包：

```text
easyopd/methods/ropd/
```

面向用户运行的 shell 入口、可直接执行的数据准备工具和 README 进入执行目录：

```text
examples/ropd_trainer/
```

这样使用者能通过 `examples/ropd_trainer` 找到完整运行链路，而库代码不会散落在 `examples` 中。

### 2. 包内资源优先

ROPD prompts 默认作为包内资源管理：

```text
easyopd/methods/ropd/prompts/
```

代码通过包内资源读取默认 prompt。若后续需要用户覆盖 prompt，可以在 `examples/ropd_trainer/prompts/` 放模板或说明，但第一阶段不让核心逻辑依赖仓库根目录的 `prompts/`。

### 3. 数据只迁移工具，不迁移产物

源仓库中的 `data/`、`datasets/` 混合了格式化工具、原始数据快照、统一数据输出和实验产物。迁移时只迁移必要的数据处理代码和 README，不迁移大数据文件或生成产物。

默认数据路径通过 `DATA_ROOT`、命令行参数或 README 中的示例指定。

### 4. 直接替换优先，不主张兼容 alias

迁移时优先把历史配置、导入路径、脚本变量和代码分支直接改写为 `ropd`，而不是新增兼容 alias、wrapper 脚本或 shim 模块。历史配置或代码里的 `shared-rubrics` / `shared_rubrics` 在迁移时直接改写为 `ropd`，不设置新的 `SHARED_RUBRICS_*` 环境变量，不创建 `shared_rubrics` 配置组。

原则是：能直接替换的地方直接替换；不能直接替换的地方先记录为迁移阻塞点或后续人工确认项。只有在确有仍需运行的外部历史作业、且直接替换会造成不可接受断裂时，才单独评估非常窄的临时兼容层。该兼容层不作为默认方案，也不作为目标结构的一部分。

## 目标目录结构

迁移完成后，建议新增或调整为：

```text
easyopd/
  config/
    ropd/
      base.yaml
      judge.yaml
      sft.yaml
  methods/
    ropd/
      __init__.py
      pipeline.py
      reward_manager.py
      clients.py
      prompts.py
      judge/
        __init__.py
        circuit_breaker.py
        config.py
        openai_env.py
        provider.py
        rate_limit.py
        resolver.py
        runtime.py
        runtime_builder.py
        scheduler.py
        schema.py
        teacher_client.py
      utils/
        __init__.py
        eval_package.py
      prompts/
        rubricator.txt
        rubricator_cn.txt
        verifier.txt
        verifier_cn.txt
        verifier_skywork.txt

examples/
  ropd_trainer/
    README.md
    train_ropd.sh

docs/
  algo/
    ropd.md
  superpowers/
    specs/
      2026-05-26-ropd-migration-to-easyopd-design.md

tests/
  easyopd/
    ropd/
      test_pipeline.py
      test_reward_manager.py
      test_clients.py
      test_judge_config.py
      test_judge_provider.py
      test_prompt_resources.py
      test_entrypoints.py
      test_no_legacy_names.py
```

## 模块迁移映射

| 源路径 | 目标路径 | 策略 |
|---|---|---|
| `algo/pipeline.py` | `easyopd/methods/ropd/pipeline.py` | 直接迁移并改 import |
| `algo/reward_manager.py` | `easyopd/methods/ropd/reward_manager.py` | 迁移为主线 reward manager |
| `algo/clients.py` | `easyopd/methods/ropd/clients.py` | 迁移 judge client 构造逻辑 |
| `algo/prompts.py` | `easyopd/methods/ropd/prompts.py` | 改为读取包内 prompt 资源 |
| `algo/prompt_utils.py` | `easyopd/methods/ropd/prompt_utils.py` | 如仍需要则迁移 |
| `algo/judge/*` | `easyopd/methods/ropd/judge/*` | 迁移 provider/runtime/schema/scheduler |
| `algo/utils/*` | `easyopd/methods/ropd/utils/*` | 迁移通用工具 |
| `prompts/custom/*.txt` | `easyopd/methods/ropd/prompts/*.txt` | 作为包内默认资源 |
| `training/ppo/train_ropd.sh` | `examples/ropd_trainer/train_ropd.sh` | 作为唯一保留的 shell 入口 |
| `tests/algo/*` | `tests/easyopd/ropd/*` | 改 import 和断言 |
| `tests/training/*ropd*` | `tests/easyopd/ropd/test_entrypoints.py` | 保留 dry-run 合同 |

## 配置设计

新增配置目录：

```text
easyopd/config/ropd/
```

建议配置分层：

- `base.yaml`：ROPD 默认训练配置，选择 ROPD 主线。
- `judge.yaml`：rubricator/verifier、provider resolution、quality gate、request scheduler 等主线参数。

入口脚本通过 Hydra override 或 EasyOPD 现有配置方式组合这些配置，不直接把大量默认值硬编码在 shell 里。

配置命名策略：

- 新配置只使用 `ropd`。
- 旧配置中的 `black_opd` 字段优先直接改成 `ropd` 字段；不在 reward manager 中默认增加兼容 alias 分支。
- 源仓库中的 `shared-rubrics` / `shared_rubrics` 命名迁移后统一改为 `ropd`，不作为 alias 继续暴露。
- `BLACK_OPD_*` 环境变量不作为目标入口的默认支持项；目标入口直接读取 `ROPD_*`。如发现必须保留的外部历史任务，再单独补充受限兼容设计。

## 执行入口设计

唯一保留的一线 shell 运行命令集中在：

```text
examples/ropd_trainer/
```

建议入口：

```bash
bash examples/ropd_trainer/train_ropd.sh
```

入口脚本约束：

- `train_ropd.sh` 从自身路径解析 `PROJECT_ROOT`，可从任意 cwd 调用。
- `train_ropd.sh` 支持 `ROPD_DRYRUN=true`，用于 CPU 合同测试和用户预检。
- `train_ropd.sh` 支持 `ROPD_SKIP_REPO_DOTENV=true`，避免测试读本地密钥。
- `train_ropd.sh` 负责加载 `.env`、wandb 目录和路径默认值；变量读取直接使用 `ROPD_*`，不新增旧环境变量 alias 脚本。
- Python 逻辑下沉到 `easyopd.methods.ropd`，shell 只保留 `train_ropd.sh` 这一层编排。

## Prompt 与资源设计

默认 prompt 存放在：

```text
easyopd/methods/ropd/prompts/
```

迁移的初始资源包括：

- `rubricator.txt`
- `rubricator_cn.txt`
- `verifier.txt`
- `verifier_cn.txt`
- `verifier_skywork.txt`

读取方式：

- Python 使用 `importlib.resources` 读取包内资源。
- 支持环境变量或配置指定外部 prompt 路径，以便实验覆盖。
- 外部路径覆盖不应成为默认行为。

## 数据与实验工具设计

源仓库的数据相关内容分三类处理：

1. 保留为代码的工具：
   - 与 `train_ropd.sh` 直接相关的最小运行辅助代码
   - 必要的 dataset format helpers

2. 保留为文档说明的资源：
   - 数据下载来源
   - 数据目录约定
   - 预处理命令
   - checksum 或 schema 说明

3. 不迁移的产物：
   - `datasets/unified/*` 下的生成数据
   - benchmark 输出
   - pool/eval 中间产物
   - wandb/log/checkpoint/cache

实验评估脚本如 `experiments/eval/eval_black_opd.py` 不在第一阶段强行主线化。若需要，应作为后续单独 spec：把评估工具改名为 ROPD eval 并迁入 `examples/ropd_trainer/tools` 或 `easyopd/methods/ropd/eval`。
实验评估、压测、计费观测和 attention backend 自动解析脚本都不在第一阶段迁移范围内。

## 文档设计

新增：

```text
docs/algo/ropd.md
examples/ropd_trainer/README.md
```

`docs/algo/ropd.md` 说明：

- ROPD 的方法定义。
- rubric-based reward 的训练流程。
- teacher/rubricator/verifier 三类 judge provider。
- 配置入口和关键环境变量。

`examples/ropd_trainer/README.md` 说明：

- 最小 dry-run 命令。
- 训练命令。
- 常用环境变量。
- 不迁移数据产物的原因和推荐 `DATA_ROOT` 布局。

`docs/superpowers/specs/` 保存本设计，用于后续写 implementation plan。

## 测试与验收

### 静态和导入检查

迁移后应通过：

```bash
python -m compileall easyopd/methods/ropd
```

并确保不存在核心路径旧导入：

```bash
rg -n "from algo|import algo|prompts/|training/ppo|black-opd" easyopd/methods/ropd examples/ropd_trainer tests/easyopd/ropd
```

允许测试中出现旧命名残留检查，但不应新增 `BLACK_OPD_*` alias 合同测试。

### 单元测试

重点迁移并通过：

```bash
uv run --no-sync pytest tests/easyopd/ropd -v
```

第一阶段至少覆盖：

- `pipeline` raw prompt 归一化。
- ROPD reward tensor 和 extra info。
- judge provider config / resolver。
- prompt resource loading。
- entrypoint dry-run。
- 旧命名残留检查，确保目标侧入口、配置和核心代码不依赖 `BLACK_OPD_*` 或 `shared_rubrics`。

### 入口 dry-run

至少通过：

```bash
ROPD_DRYRUN=true ROPD_SKIP_REPO_DOTENV=true bash examples/ropd_trainer/train_ropd.sh
```

dry-run 需要打印：

- 解析后的 `PROJECT_ROOT`
- 使用的 config 名称
- 数据路径
- 模型路径来源
- judge provider 来源
- 最终 Python/Hydra 命令

## 风险与缓解

### 风险 1：旧路径导入太多，迁移后容易漏改

缓解：

- 先迁移核心包，再全仓 `rg "from algo|import algo"`。
- 不默认提供 compatibility shim；发现漏改时优先修正为直接 `easyopd.methods.ropd` 导入。
- 若确有外部历史作业需要旧路径，应先在 spec/plan 中单独列出原因、范围和移除时间，再决定是否加临时兼容层。

### 风险 2：prompt 相对路径变化导致运行时找不到资源

缓解：

- 使用 `importlib.resources` 读取包内 prompt。
- 增加 `test_prompt_resources.py`。
- 入口 dry-run 打印 prompt 来源。

### 风险 3：shell 脚本硬编码源仓库层级

缓解：

- `train_ropd.sh` 从 `BASH_SOURCE[0]` 解析 `PROJECT_ROOT`。
- 将 `_paths.sh` 能力内联或迁移为 `examples/ropd_trainer/_env.sh`。
- 对 `train_ropd.sh` 写 subprocess dry-run 测试。

### 风险 4：大数据和运行产物误入迁移

缓解：

- 迁移前生成 exclude list。
- 使用 `find` / `git status` 检查新增文件。
- 不复制 `datasets/unified`、`outputs`、`logs`、`wandb`、缓存目录。

### 风险 5：EasyOPD 与 black-opd 依赖版本不一致

缓解：

- 不覆盖 EasyOPD 的依赖文件。
- 先跑 CPU 单测，遇到缺依赖再最小增量处理。
- 对只用于可选工具的依赖，优先文档化或放 optional extra，不污染核心依赖。

## 实施切分建议

### Phase 1：迁移核心方法包

- 新建 `easyopd/methods/ropd/`。
- 迁移 `pipeline`、`reward_manager`、`clients`、`judge` 的最小闭环。
- 修改 import。
- 建立 `tests/easyopd/ropd` 的核心单测。

验收：核心 import 和 reward manager 单测通过。

### Phase 2：迁移 prompt 和配置

- 迁移包内 prompt。
- 新建 `easyopd/config/ropd/`。
- reward manager 支持新配置键。
- 增加 prompt resource 和 config smoke 测试。

验收：配置可组合，prompt 可加载。

### Phase 3：迁移执行入口

- 新建 `examples/ropd_trainer/`。
- 只迁移 `train_ropd.sh` 作为目标侧唯一 shell 入口。
- 建立 `_env.sh` 或等价环境解析 helper。
- 加 dry-run 测试。

验收：`train_ropd.sh` 的 dry-run 命令通过。

### Phase 4：文档与清理

- 新增 `docs/algo/ropd.md`。
- 新增 `examples/ropd_trainer/README.md`。
- 全仓检查旧路径残留。
- 确认没有运行产物被引入。

验收：文档路径完整，`git status` 只包含预期文件。

## 开放问题

1. ROPD 配置主入口是否使用 `ropd/base`，以及 `judge.yaml` 是否作为同一组下的组合配置。
2. `experiments/eval` 是否属于本次迁移范围，还是后续单独迁移？
3. 是否需要在 EasyOPD 根 README 中加入 ROPD 入口，还是只更新 `examples/ropd_trainer/README.md`？

## 完成定义

本次迁移完成时，应满足：

- ROPD 核心代码位于 `easyopd/methods/ropd/`。
- 用户入口位于 `examples/ropd_trainer/`。
- 默认 prompt 是包内资源。
- 配置位于 `easyopd/config/ropd/`。
- 测试位于 `tests/easyopd/ropd/`。
- 不存在误迁移的运行产物、缓存或大数据集。
- `ROPD_DRYRUN=true` 的核心入口可执行到命令生成。
- 目标侧只保留 `train_ropd.sh` 这一条 shell 入口。
- 目标侧不迁移 `build_index` 相关工具和 shell 入口。
- 目标入口直接使用 `ROPD_*`；不默认提供 `BLACK_OPD_*` alias 脚本。
- 目标侧不存在 `shared_rubrics` 配置组、目录或环境变量；源仓库该语义已统一映射为 `ropd`。
- 目标侧不新增 wrapper 脚本、shim 模块或兼容 alias 作为迁移主路径。
- ROPD 主线之外的 ablation 没有被迁入目标仓库。
- 中文文档说明清楚迁移范围、运行方式和不迁移内容。
