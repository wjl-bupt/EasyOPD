
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

每个方法在 `easyopd/methods/` 下有一个独立子目录，结构如下：

```
easyopd/methods/<方法名>/
├── __init__.py          # 导入并注册方法
├── trainer.py           # Trainer 类（继承或组合 verl 的 trainer）
├── losses.py            # 方法特有的 loss 函数（如有）
├── reward_manager.py    # 方法特有的 reward 管理（如有）
├── utils.py             # 方法内部工具函数（如有）
└── README.md            # 方法说明文档
```

对应的配置文件放在：

```
easyopd/config/<方法名>.yaml
```

对应的训练脚本放在：

```
examples/<方法名>/
├── run_xxx.sh           # 训练启动脚本
└── README.md            # 使用说明
```

### 关键原则

1. **不要随意修改 `verl/` 目录下的代码**。如果确实需要修改 verl 底层代码，必须在 commit message 中说明原因。
2. **方法之间互不依赖**。你的方法代码应该完全在 `easyopd/methods/<你的方法>/` 内自包含。
3. **公共工具放 `easyopd/utils/`**。如果你写了一个多个方法都可能用到的工具函数，放到 `easyopd/utils/` 下。

---

## 6. 方法接入规范

### 6.1 注册你的方法

在 `easyopd/registry.py` 中使用装饰器注册：

```python
# easyopd/methods/simct/__init__.py
from easyopd.registry import register_method

@register_method("simct")
class SimCTTrainer:
    """SimCT: Cross-Tokenizer On-Policy Distillation"""

    def __init__(self, config):
        ...

    def train(self):
        ...
```

### 6.2 编写 yaml 配置文件

```yaml
# easyopd/config/simct.yaml
method:
  name: simct
  description: "SimCT: Recovering Lost Supervision for Cross-Tokenizer OPD"

model:
  student_model_path: "google/gemma-2-2b-it"
  teacher_model_path: "Qwen/Qwen2.5-7B-Instruct"

training:
  num_epochs: 3
  batch_size: 128
  learning_rate: 5e-6
  # ... 其他训练参数

distillation:
  # 方法特有的蒸馏参数
  loss_mode: "simct"
  # ...
```

### 6.3 用户调用方式（目标接口）

```python
from easyopd import EasyOPD

# 一行调用
trainer = EasyOPD.from_hparams("simct", config_path="easyopd/config/simct.yaml")
trainer.train()
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

### Q5: 需要修改 verl 底层代码怎么办？

1. 先确认是否真的需要改（能否在 `easyopd/` 层面解决）
2. 如果必须改，在 commit message 中清楚说明原因
3. 尽量做最小改动，不要大范围重构 verl 代码
4. 在 PR 描述中标注修改了 verl 的哪些文件

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
