
# EasyOPD 项目协作说明

EasyOPD 是一个基于 [verl](https://github.com/volcengine/verl) 的统一 On-Policy Distillation (OPD) 框架。我们的目标是将 ROPD、SimCT、SOD、DSKD、ALM 等不同的 OPD 方法整合到一个统一的代码库中，提供 EasyEdit 式的统一用户接口，让用户只需切换 yaml 配置文件即可调用不同的 OPD 方法。

- **项目地址：** https://github.com/lds-ustc/EasyOPD.git
- **底层框架：** verl（原始 verl 文档见 [README_VERL.md](README_VERL.md)）
- **待整合的方法仓库：**
  - [ROPD](https://github.com/Peregrine123/ROPD_official) — Rubric-based On-Policy Distillation（黑盒 OPD）
  - [SimCT](https://github.com/sunjie279/SimCT-) — Cross-Tokenizer On-Policy Distillation
  - [SOD](https://github.com/YoungZ365/SOD) — Step-wise On-policy Distillation（Tool-Integrated Reasoning 场景）
  - 以及 DSKD、ALM、ULD、GKD 等其他 OPD 方法

---

## 目录

- [1. 架构概览](#1-架构概览)
- [2. 克隆项目](#2-克隆项目)
- [3. 环境安装](#3-环境安装)
- [4. 创建自己的开发分支](#4-创建自己的开发分支)
- [5. 代码目录结构规范](#5-代码目录结构规范)
- [6. 方法接入规范](#6-方法接入规范)
- [7. 开发与提交流程](#7-开发与提交流程)
- [8. 同步上游更新与冲突处理](#8-同步上游更新与冲突处理)
- [9. 完整命令示例](#9-完整命令示例)
- [10. 常见问题](#10-常见问题)
- [11. 注意事项](#11-注意事项)

---

## 1. 架构概览

EasyOPD 的核心思路是：**底层架构沿用 verl，在其上做一层统一用户接口**。

```
EasyOPD/
├── verl/                          # verl 底层框架（尽量少改）
│   ├── trainer/
│   │   ├── distillation/          # verl 官方 OPD 基础设施
│   │   ├── main_ppo.py            # verl 官方训练入口
│   │   └── ...
│   └── ...
│
├── easyopd/                       # ★ EasyOPD 统一接口层（核心差异化）
│   ├── __init__.py                # from_hparams() 一行调用入口
│   ├── registry.py                # 方法注册表
│   ├── config/                    # 统一 yaml 配置（每个方法一个）
│   │   ├── gkd.yaml
│   │   ├── ropd.yaml
│   │   ├── simct.yaml
│   │   ├── sod.yaml
│   │   └── ...
│   ├── methods/                   # 各方法实现（独立子目录，互不冲突）
│   │   ├── gkd/
│   │   ├── ropd/
│   │   ├── simct/
│   │   ├── sod/
│   │   ├── dskd/
│   │   └── ...
│   └── utils/                     # 公共工具
│
├── examples/                      # 训练脚本
├── recipe/                        # verl-recipe submodule
├── README.md                      # 本文件
├── README_VERL.md                 # verl 原始 README
└── ...
```

**差异化定位：** 与 verl 的差异不在底层算法或并行架构上，而在"易用性"和"统一性"上。对用户而言，调用任何一个 OPD 方法都是同一套 API，只需切换 yaml；对内部而言，每个方法仍然是独立子目录，便于各自维护和扩展。

---

## 2. 克隆项目

### 方式一：HTTPS（推荐，集群环境通用）

```bash
git clone https://github.com/lds-ustc/EasyOPD.git
cd EasyOPD
```

如果需要 push 权限，使用 Personal Access Token：

```bash
git clone https://<your-github-token>@github.com/lds-ustc/EasyOPD.git
cd EasyOPD
```

> **Token 生成方式：** GitHub → Settings → Developer settings → Personal access tokens → Generate new token，勾选 `repo` 权限。

### 方式二：SSH

```bash
git clone git@github.com:lds-ustc/EasyOPD.git
cd EasyOPD
```

> ⚠️ 很多集群环境会封锁 SSH 的 22 端口，如果报 `Network is unreachable` 或 `Connection refused`，请改用 HTTPS 方式。

### 初始化 submodule

```bash
git submodule update --init
```

这会拉取 `recipe/` 目录下的 verl-recipe 子模块。

---

## 3. 环境安装

### 3.1 基础环境要求

- Python >= 3.10
- CUDA >= 12.1（推荐 12.6）
- PyTorch >= 2.4

### 3.2 安装 verl 及依赖

```bash
# 以开发模式安装（推荐）
pip install -e .

# 安装 flash-attn（需要 CUDA 环境）
pip install flash-attn --no-build-isolation

# 如果需要 vLLM 推理引擎
pip install vllm
```

### 3.3 验证安装

```bash
python -c "import verl; print(verl.__version__)"
```

---

## 4. 创建自己的开发分支

### ⚠️ 重要：基线版本是 main 分支，不是 v0.7.1

因为 **verl 的 OPD 功能（`verl/trainer/distillation/`）是在 v0.7.1 之后才合入 main 的**，v0.7.1 上完全没有 distillation 相关代码。所以所有人必须基于 **main 分支** 开发。

我们锁定 main 分支的以下 commit 作为统一基线：

```
802256a7 [doc] chore: OPD docs (#6358)
```

### 创建分支

```bash
# 确保在最新的 main 上
git checkout main
git pull origin main

# 基于 main 创建自己的开发分支
git checkout -b <你的名字>-<方法名> main
```

### 分支命名规范

格式：`姓名拼音-方法名`，例如：

```bash
git checkout -b zhangsan-ropd main
git checkout -b lisi-simct main
git checkout -b wangwu-dskd main
git checkout -b zhaoliu-alm main
```

### 确认当前分支

```bash
git branch
# 当前分支前面会有 *，例如：
# * zhangsan-ropd
#   main
```

---

## 5. 代码目录结构规范

### 5.1 整体目录结构

```
EasyOPD/
├── verl/                          # verl 底层（允许按规范修改）
├── easyopd/                       # ★ EasyOPD 统一接口层
│   ├── __init__.py                # from_hparams() 入口
│   ├── registry.py                # 方法注册表
│   ├── config/                    # 统一 yaml 配置
│   │   ├── gkd.yaml
│   │   ├── ropd.yaml
│   │   ├── simct.yaml
│   │   ├── sod.yaml
│   │   └── ...
│   ├── methods/                   # 各方法的核心实现
│   │   ├── gkd/
│   │   ├── ropd/
│   │   ├── simct/
│   │   ├── sod/
│   │   ├── dskd/
│   │   └── ...
│   └── utils/                     # 公共工具
├── examples/                      # 训练脚本（每个方法一套）
│   ├── ropd/
│   ├── simct/
│   ├── sod/
│   └── ...
└── ...
```

### 5.2 方法实现的三种模式

根据对已有方法的调研，不同方法对 verl 的修改程度不同，大致分为三种模式：

#### 模式 A：轻量修改 verl 配置 + 在 trainer 中加逻辑（如 SOD）

SOD 的做法是在 verl 的配置 dataclass 中加几个字段，在 `ray_trainer.py` 中加一段 if 分支逻辑。这种方式改动最小，但直接嵌入了 verl 代码。

**接入方式：**

1. 将你的核心算法逻辑（如 `compute_stepwise_opd_weights` 函数）抽出来放到 `easyopd/methods/sod/core.py`
2. 在 verl 的 trainer 中通过 **import 调用** 你的函数，而不是把函数体直接写在 verl 文件里
3. 配置字段可以加在 verl 的 config dataclass 中（因为 verl 的 hydra 配置系统需要）

```
easyopd/methods/sod/
├── __init__.py
├── core.py              # compute_stepwise_opd_weights() 等核心算法
└── README.md            # 方法说明、参数含义、复现步骤

verl/trainer/config/algorithm.py   # 加 stepwise_* 配置字段（最小改动）
verl/trainer/ppo/ray_trainer.py    # 加 if stepwise_enable 分支，import sod.core

examples/sod/
├── run_sod.sh           # 训练脚本
└── README.md
```

#### 模式 B：独立的 reward/pipeline 模块 + 修改 verl 入口（如 ROPD）

ROPD 是黑盒 OPD，需要独立的 rubric 生成和评分 pipeline，同时修改了 verl 的 reward_manager 和 fully_async 模块。

**接入方式：**

1. 独立的 pipeline 代码放 `easyopd/methods/ropd/`（如 rubricator、verifier、judge worker）
2. 对 verl 的修改集中在 reward_manager 注册和 trainer 入口

```
easyopd/methods/ropd/
├── __init__.py
├── pipeline.py          # ROPD 主 pipeline（rubric 生成 → 评分 → reward）
├── reward_manager.py    # ROPD 的 reward manager
├── judge_worker.py      # Judge 模型 worker
├── prompts/             # Rubric/Verify prompt 模板
└── README.md

verl/workers/reward_manager/  # 注册 ROPD reward manager

examples/ropd/
├── run_ropd.sh
├── launch_judge_vllm.sh
└── README.md
```

#### 模式 C：完全独立的训练框架，需要重新适配（如 SimCT/KDFlow）

SimCT 原本基于 KDFlow（独立框架），有自己的 trainer、dataset、loss 体系，需要在 verl 框架下重新实现。

**接入方式：**

1. 核心算法（cross-tokenizer alignment、loss 函数）放 `easyopd/methods/simct/`
2. 利用 verl 已有的 distillation 基础设施（`verl/trainer/distillation/`），通过扩展 loss_mode 接入
3. 如果 verl 的 distillation 模块不支持你的场景（如跨 tokenizer），可以在 verl 的 loss 注册表中添加新的 loss 类型

```
easyopd/methods/simct/
├── __init__.py
├── losses.py            # SimCT 特有的 cross-tokenizer loss
├── alignment.py         # tokenizer 对齐工具
├── trainer.py           # 如果需要自定义 trainer 逻辑
└── README.md

verl/trainer/distillation/losses.py  # 注册 simct loss mode（最小改动）

examples/simct/
├── run_simct.sh
└── README.md
```

### 5.3 关键原则

1. **核心算法逻辑放 `easyopd/methods/<方法名>/`**。即使你需要改 verl 文件，也要把算法的核心实现（loss 函数、权重计算、pipeline 逻辑等）放在自己的目录下，verl 文件中只做 import 和调用。
2. **对 verl 的修改要最小化、可追踪**。改 verl 文件时：
   - 用注释标明 `# [EasyOPD] Added for <方法名>`
   - 尽量用 if 分支或注册机制，不要删改原有逻辑
   - commit message 中列出改了 verl 的哪些文件
3. **方法之间不要互相依赖**。你的 if 分支不应该影响别人的 if 分支。
4. **每个方法必须有 README.md**，说明：
   - 方法原理简述（一段话）
   - 修改了 verl 的哪些文件、为什么要改
   - 如何运行（完整的复现步骤）
   - 依赖的模型和数据

### 5.4 对 verl 文件的修改规范

由于很多方法不可避免要改 verl 的文件，为了避免冲突和混乱，请遵循：

```python
# ============ [EasyOPD:SOD] Step-wise OPD weighting ============
# 在这里添加你的逻辑
stepwise_enable = getattr(cfg, "stepwise_enable", False)
if stepwise_enable:
    from easyopd.methods.sod.core import compute_stepwise_opd_weights
    # ... 调用你的方法
# ============ [EasyOPD:SOD] End ============
```

**规范要点：**
- 用 `# [EasyOPD:<方法名>]` 注释包裹你的修改
- 新增的配置字段要有默认值（`False` / `None` / `0`），确保不开启时不影响原有行为
- 不要删除或修改 verl 原有的代码逻辑，只做**增量添加**

---

## 6. 方法接入规范

### 6.1 注册你的方法

在 `easyopd/methods/<方法名>/__init__.py` 中使用装饰器注册：

```python
# easyopd/methods/sod/__init__.py
from easyopd.registry import register_method

@register_method("sod")
class SODMethod:
    """SOD: Step-wise On-policy Distillation"""

    # 方法元信息
    verl_modified_files = [
        "verl/trainer/config/algorithm.py",      # 添加 stepwise_* 配置
        "verl/trainer/ppo/ray_trainer.py",        # 添加 stepwise OPD 分支
    ]

    def __init__(self, config):
        ...

    def train(self):
        ...
```

### 6.2 编写 yaml 配置文件

配置文件应该能让用户**一键复现**你的实验：

```yaml
# easyopd/config/sod.yaml
method:
  name: sod
  description: "SOD: Step-wise On-policy Distillation for Small LM Agents"

model:
  student_model_path: "Qwen/Qwen3-1.7B"
  teacher_model_path: "<GRPO-optimized teacher checkpoint>"

training:
  # verl 标准训练参数
  adv_estimator: grpo
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28

  # SOD 特有参数
  token_kl_reg:
    stepwise_enable: true
    stepwise_epsilon: 1e-6
    stepwise_delta: 0.5
    stepwise_opd_coef: 1.0

data:
  train_files: ["<path to RL training data>"]
  test_files: ["<path to eval data>"]
```

### 6.3 编写方法 README

每个方法的 `easyopd/methods/<方法名>/README.md` 必须包含：

```markdown
# <方法名>

## 方法简介
一段话描述方法的核心思想。

## 对 verl 的修改
| 文件 | 修改内容 | 原因 |
|------|----------|------|
| verl/trainer/config/algorithm.py | 添加 stepwise_* 字段 | SOD 需要配置步级权重参数 |
| verl/trainer/ppo/ray_trainer.py | 添加 compute_stepwise_opd_weights 调用 | 核心算法入口 |

## 复现步骤
1. 数据准备：...
2. 模型准备：...
3. 运行训练：`bash examples/sod/run_sod.sh`
4. 评测：...

## 实验结果
| Benchmark | Score |
|-----------|-------|
| ... | ... |
```

### 6.4 用户调用方式（目标接口）

```python
from easyopd import EasyOPD

# 一行调用
trainer = EasyOPD.from_hparams("sod", config_path="easyopd/config/sod.yaml")
trainer.train()
```

或者直接用 verl 的方式运行（因为方法已经注入到 verl 中）：

```bash
# 直接用训练脚本
bash examples/sod/run_sod.sh
```

---

## 7. 开发与提交流程

### 7.1 开发前确认分支

```bash
git branch
# 确保不在 main 分支上！
```

如果发现在 main 上，先切换：

```bash
git checkout <你的名字>-<方法名>
```

### 7.2 查看修改

```bash
# 查看修改了哪些文件
git status

# 查看具体改了什么
git diff

# 查看某个文件的修改
git diff path/to/file.py
```

### 7.3 提交 commit

```bash
# 添加文件
git add easyopd/methods/simct/
git add easyopd/config/simct.yaml
git add examples/simct/

# 提交（commit message 要清晰）
git commit -m "Add SimCT method: trainer, losses, and config"
```

**Commit message 规范：**

```bash
# ✅ 好的 commit message
git commit -m "Add SimCT training pipeline and cross-tokenizer alignment"
git commit -m "Implement ROPD rubric-based reward manager"
git commit -m "Fix tokenizer alignment bug in SimCT loss computation"
git commit -m "Add DSKD config and example scripts"

# ❌ 不好的 commit message
git commit -m "update"
git commit -m "fix"
git commit -m "test"
```

### 7.4 推送到远端

```bash
# 第一次推送
git push -u origin <你的名字>-<方法名>

# 之后直接
git push
```

### 7.5 不要提交的文件

以下文件已在 `.gitignore` 中配置，不会被提交：

- `*.pt` / `*.bin` / `*.safetensors` / `*.ckpt` — 模型权重
- `checkpoints/` / `outputs/` / `wandb/` — 训练产物
- `__pycache__/` — Python 缓存
- `*.log` — 日志文件

如果你发现有大文件被 `git add` 了，请在提交前移除：

```bash
git reset HEAD path/to/large_file.bin
```

---

## 8. 同步上游更新与冲突处理

### 8.1 拉取最新的 main 分支

当项目管理者修复了公共 bug 或更新了公共代码时，你需要同步：

```bash
# 先切到 main 拉取最新
git checkout main
git pull origin main

# 切回自己的分支
git checkout <你的名字>-<方法名>

# 将 main 的更新合并到自己的分支
git merge main
```

### 8.2 处理合并冲突

如果 merge 时出现冲突：

```bash
# 查看哪些文件有冲突
git status

# 手动编辑冲突文件，解决冲突标记（<<<<<<< / ======= / >>>>>>>）

# 解决完后标记为已解决
git add <冲突文件>
git commit -m "Merge main and resolve conflicts"
```

**减少冲突的最佳实践：**
- 不要修改 `verl/` 下与自己方法无关的代码
- 经常同步 main（不要等到最后才 merge）
- 方法代码放在自己的独立目录下

### 8.3 使用 rebase（可选，适合熟悉 git 的同学）

```bash
git checkout <你的名字>-<方法名>
git fetch origin
git rebase origin/main
# 如果有冲突，逐个解决后 git rebase --continue
git push --force-with-lease
```

---

## 9. 完整命令示例

假设你的名字是 zhangsan，要集成 ROPD 方法：

```bash
# 1. 克隆项目
git clone https://<your-token>@github.com/lds-ustc/EasyOPD.git
cd EasyOPD

# 2. 初始化 submodule
git submodule update --init

# 3. 安装环境
pip install -e .
pip install flash-attn --no-build-isolation

# 4. 创建开发分支
git checkout -b zhangsan-ropd main

# 5. 创建方法目录结构
mkdir -p easyopd/methods/ropd
mkdir -p easyopd/config
mkdir -p examples/ropd

# 6. 开发你的方法...
#    - 实现 easyopd/methods/ropd/trainer.py
#    - 编写 easyopd/config/ropd.yaml
#    - 编写 examples/ropd/run_xxx.sh

# 7. 检查修改
git status
git diff

# 8. 提交
git add easyopd/methods/ropd/
git add easyopd/config/ropd.yaml
git add examples/ropd/
git commit -m "Add ROPD method: rubric-based reward manager and trainer"

# 9. 推送
git push -u origin zhangsan-ropd
```

---

## 10. 常见问题

### Q1: SSH clone 报错 `Network is unreachable`

集群环境通常封锁了 SSH 端口，改用 HTTPS 方式：

```bash
git clone https://<your-token>@github.com/lds-ustc/EasyOPD.git
```

### Q2: push 时提示输入用户名密码

使用 HTTPS 方式时需要 token。可以将 token 配置到 git 中避免每次输入：

```bash
git remote set-url origin https://<your-token>@github.com/lds-ustc/EasyOPD.git
```

### Q3: 不小心在 main 分支上改了代码

不要慌，先创建新分支保留修改：

```bash
git checkout -b <你的名字>-<方法名>
git add .
git commit -m "Save local changes"
```

然后恢复 main：

```bash
git checkout main
git reset --hard origin/main
```

### Q4: pull 时提示本地修改会被覆盖

先提交或暂存你的修改：

```bash
# 方式一：提交
git add .
git commit -m "Save work in progress"

# 方式二：暂存
git stash
# pull 之后再恢复
git stash pop
```

### Q5: 我的方法需要修改 verl 的文件怎么办？

这是正常的，很多方法都需要改 verl 文件（参见第 5.2 节的三种模式）。请遵循：

1. 核心算法逻辑放 `easyopd/methods/<你的方法>/`，verl 中只做 import 调用
2. 用 `# [EasyOPD:<方法名>]` 注释包裹你的修改
3. 新增配置字段要有默认值，确保不开启时不影响原有行为
4. 不要删改 verl 原有逻辑，只做增量添加
5. 在方法 README 中列出你改了 verl 的哪些文件

### Q6: `recipe/` 目录是空的

需要初始化 submodule：

```bash
git submodule update --init
```

---

## 11. 注意事项

### 三条铁律

1. **不要直接在 main 分支上开发。** 所有开发都在自己的分支上进行。
2. **所有人基于 main 分支创建自己的分支。** 统一基线，保证后续可以公平比较和合并。
3. **提交前必须用 `git status` 和 `git diff` 检查修改内容。** 避免提交无关文件或大文件。

### 代码规范

- 项目配置了 `pre-commit` hooks（ruff 格式化、mypy 类型检查等），建议安装：
  ```bash
  pip install pre-commit
  pre-commit install
  ```
- 如果不想安装 pre-commit，至少在提交前手动检查格式：
  ```bash
  pip install ruff
  ruff check --fix .
  ruff format .
  ```
- 注意命名：使用 `verl`（小写），不要写 `veRL`。

### 实验记录

- 每个方法的 `examples/<方法名>/README.md` 中应记录：
  - 使用的 teacher/student 模型
  - 训练超参数
  - 评测 benchmark 和结果
- 统一评测 benchmark：GSM8K、MATH-500、MBPP、LiveCodeBench（具体以最终讨论为准）
