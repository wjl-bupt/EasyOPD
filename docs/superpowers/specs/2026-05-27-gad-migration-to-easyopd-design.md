# GAD 迁移到 EasyOPD 的总体设计

## 背景

当前涉及三个仓库:

- 目标仓库:`/mnt/d/Area/DL/projects/research/EasyOPD`(本仓库;当前分支 `zhepei-GAD`,基于 `main`)
- 源仓库(驱动层):`microsoft/LMOps/gad`,本地参考拷贝在 `/mnt/d/Area/DL/projects/research/LMOps-gad/`(无 git 历史,不被 EasyOPD 跟踪)
- 源仓库(算法层):`YTianZHU/verl` 的 `gad` 分支,本地参考拷贝在 `/mnt/d/Area/DL/projects/research/YTianZHU-verl-gad/`(无 git 历史,不被 EasyOPD 跟踪)

EasyOPD 是基于 verl 的 OPD 方法集合仓库,已通过 `easyopd/methods/simple`、`easyopd/methods/simct`、`easyopd/methods/sod` 等方法目录建立了模式,并以 `zhepei/ropd` 分支跑通了一次完整的"独立方法包 + verl 最小入侵"集成。

GAD(Generative Adversarial Distillation,论文 [arXiv:2511.10643](https://arxiv.org/abs/2511.10643))在源仓库实现里把 verl 的 `critic` 模块改造为 discriminator,用 Bradley-Terry pairwise loss 区分 student 与 teacher response;discriminator 给出的分数被注入 actor 的 advantage 计算,从而以纯黑盒方式实现 on-policy distillation。论文方法分为 SeqKD baseline、warmup、GAD 对抗、eval-only 四阶段,分布在源 fork 的四个分支中。

本次目标不是把 `YTianZHU/verl@gad` 原样嵌入 EasyOPD,而是把 GAD **对抗阶段**收敛成 EasyOPD 中一个正式、可维护、路径清晰的方法模块。

## 目标

1. 将 GAD 作为 EasyOPD 的一等方法迁移到 `easyopd/methods/gad/`。
2. 仅集成 GAD **对抗阶段**(`gad` 分支的算法表面),不集成 SeqKD baseline、warmup、eval-only。
3. 将配置、入口脚本、测试和文档放到 EasyOPD 现有结构能解释的位置(`easyopd/config/gad/`、`examples/gad_trainer/`、`tests/easyopd/gad/`、`docs/algo/gad.md`)。
4. **verl 主干尽量不动**:所有算法逻辑在 `easyopd/methods/gad/`,verl 文件中只有 `# [EasyOPD:GAD]` 注释包裹的 if 分支做 import 调用。
5. **不开启时零行为变化**:所有新增 config 字段默认 `enable=False`/`None`;现有 verl 测试和其它方法不受影响。
6. **不跨方法依赖**:不引用 `easyopd/methods/ropd|simct|sod|simple`。
7. 用 CPU 级合同测试证明结构可导入、可注册、契约稳定、关键纯函数数值正确,并能进入 dry-run 阶段。

## 非目标

第一阶段迁移不做以下事情:

- 不迁移 SeqKD baseline、warmup、eval-only 三个阶段(留给后续工作)。
- 不迁移 teacher 数据准备工具(`LMOps-gad/tools/export_lmsys_parquet.py`),只在 README 中指路上游。
- 不迁移 `LMOps-gad/deepscaler/`(math reward 工具集,与 GAD 对抗阶段无关)。
- 不重写 GAD 算法语义(BT loss、critic-as-discriminator 的核心思想保持不变)。
- 不实现 discriminator 的自动 warmup / 预训练 —— 用户必须提供已训好的 discriminator checkpoint。
- 不引入 ray、vllm、GPU 才能跑的运行产物或测试。
- 不修改 EasyOPD 的 `pyproject.toml`、`requirements*.txt`、`setup.py` 主结构(只在确实需要新包数据时按 ROPD 范式追加)。
- 不在第一阶段删除 EasyOPD 现有 ROPD / SOD / SimCT / Simple 功能(本分支 `zhepei-GAD` 是从 `main` 切出的,不含 `zhepei/ropd` 内容)。
- 不准备直接向 upstream `verl-project/verl` 发 PR;若未来要发,需按 AGENTS.md 做 duplicate-work checks 和人工责任说明。

## 总体迁移原则

### 1. 方法归方法,入口归入口

GAD 的 Python 主实现进入方法包:

```text
easyopd/methods/gad/
```

面向用户的 shell 入口、训练脚本和 README 进入执行目录:

```text
examples/gad_trainer/
```

### 2. verl 一侧只做"开关 + import 调用",不写算法

verl 一侧的所有改动都必须满足:

- 用 `# ============ [EasyOPD:GAD] ... # ============ [EasyOPD:GAD] End ============` 注释成对包裹。
- 分支内只做"取 config flag → import easyopd.methods.gad.<fn> → 调用",**不写算法主体**。
- 新增 config 字段或 keyword 参数必须有默认值,使得 `gad.enable=false` 时分支不进入,verl 原有行为零变化。
- 不删除、不修改 verl 现有逻辑;只做增量添加。

### 3. 直接替换优先,不主张兼容 alias

源仓库的脚本和命名(`gpt5-chat-filtered-7b-adversarial-lr1e-6.sh` 等)在迁移时直接改写为 `gad` 风格命名,不保留旧文件名作为兼容入口。源仓库的 `deepscaler` namespace 不进入 EasyOPD。

## 架构总览

GAD 的"对抗"在 verl 里的体现是 **critic 充当 discriminator**:

| verl 表面 | 是否动 |
|---|---|
| Actor loss(`dp_actor.update_policy`) | **不动** —— 标准 PPO 公式,只是 advantage 来源换了 |
| Critic loss(`dp_critic.update_critic`) | **重写** —— 从 MSE value loss 换成 BT pairwise loss |
| Critic forward(`dp_critic._forward_micro_batch`) | **改一处** —— 末 token 当 seq-level score;支持 `compute_teacher` 切 student/teacher 输入 |
| `critic.compute_values`(trainer L1086 调用) | **同 `_forward_micro_batch`** —— 输出形状从 token-value 变 last-token-only |
| Trainer fit loop(`ray_trainer.fit`) | **轻微改** —— `gen_batch` pop 列表多 `teacher_response` / `teacher_attention_mask`;rollout 后透传一次 |
| Rollout(`vllm_rollout_spmd`) | **不动** —— 在 trainer 一侧做 teacher_response 透传,避免改 rollout 文件 |
| Dataset | **数据约定** —— 训练样本必须带 `teacher_response` 字段;由 README 指路使用者准备 |

**关键设计选择:** 与上游不同,本设计 **在 trainer 一侧透传 teacher_response,而不是改 `vllm_rollout_spmd.py`**。代价是多一次小 DataProto 复制,收益是不碰 rollout 模块,未来 verl 升级 rollout 时零冲突。

**总 verl diff 预算:~31 行,集中在 2 个文件**(`verl/trainer/ppo/ray_trainer.py` 与 `verl/workers/critic/dp_critic.py`)。

## 目标目录结构

### 新增文件

```
easyopd/methods/gad/
├── __init__.py                 # @register_method("gad"); re-export 关键符号
├── core.py                     # 纯函数:BT loss、summed_reward、discriminator_accuracy、last_token_only
├── critic_forward.py           # forward_micro_batch 适配 + remap_to_teacher
├── critic_update.py            # update_critic_step(替换 dp_critic.update_critic 主体)
├── rollout_passthrough.py      # pass_teacher_response_through(在 trainer 一侧)
├── data_contract.py            # GAD_BATCH_KEYS、validate_gad_batch、GADBatchContractError
├── config.py                   # GADConfig dataclass、load_from_omegaconf、is_gad_enabled、GADConfigError
└── README.md                   # 方法说明、verl 修改清单、数据约定、复现步骤、引用论文

easyopd/config/gad/
└── base.yaml                   # Hydra 配置,defaults 链 gae + ppo

examples/gad_trainer/
├── README.md
└── train_gad.sh                # hydra 启动脚本

docs/algo/
└── gad.md                      # 方法 doc(归 docs/index.rst)

tests/easyopd/gad/
├── __init__.py
├── conftest.py                 # MockCriticModule、build_tiny_batch
├── test_imports.py
├── test_registration.py
├── test_core_numeric.py
├── test_data_contract.py
├── test_config.py
├── test_rollout_passthrough.py
├── test_critic_forward.py
├── test_critic_update_contract.py
├── test_no_drift.py            # verl 4 处 [EasyOPD:GAD] 标记完整性
├── test_no_actor_changes.py    # dp_actor.py 不含 [EasyOPD:GAD] 标记
├── test_config_smoke.py
└── test_entrypoints.py
```

### 需要小幅修改的现有文件

```
verl/trainer/ppo/ray_trainer.py        # 2 处 [EasyOPD:GAD] 注释包裹的 if 分支
verl/workers/critic/dp_critic.py       # 2 处 [EasyOPD:GAD] 注释包裹的 if 分支
docs/index.rst                         # 把 docs/algo/gad.md 接入侧栏
```

### 不修改

```
verl/workers/actor/                    # 完全不动(actor 不需要 GAD 特化)
verl/workers/rollout/                  # 完全不动(透传放在 trainer 一侧)
verl/trainer/ppo/core_algos.py         # 不动(BT loss 留在 easyopd.methods.gad.core)
其它 easyopd/methods/                  # 不引用、不依赖
pyproject.toml / setup.py              # 不动(只新增纯 Python + .txt prompts 时不需要)
```

## 组件分解

### `core.py` —— 纯函数,无副作用

- `compute_discriminator_loss(student_vpreds, teacher_vpreds, response_mask, teacher_response_mask) -> Tensor` —— BT loss:`-logsigmoid(teacher_reward − student_reward).mean()`。直接搬上游 `core_algos.compute_discriminator_loss`,语义不变。
- `summed_reward(vpreds, response_mask) -> Tensor` —— `(vpreds * mask).sum(dim=-1)`(seq-level scalar)。
- `discriminator_accuracy(student_vpreds, teacher_vpreds, student_mask, teacher_mask) -> float` —— `mean(teacher_sum > student_sum)`。
- `last_token_only(values, response_mask) -> Tensor` —— 保留每行最后一个有效位置,其它清零。

依赖:`torch`。**无 verl 依赖**,完全 CPU 可单测。

### `critic_forward.py` —— Critic forward 适配

- `forward_micro_batch(critic_module, data, *, compute_teacher: bool, use_remove_padding: bool, response_length: int, ulysses_sp_size: int) -> Tensor` —— 重写后的 micro-batch forward。当 `compute_teacher=True` 时先调 `remap_to_teacher(data)` 切换输入键。
- `remap_to_teacher(data) -> data` —— 把 `data["input_ids"]` 等键替换为 `teacher_response` / `teacher_attention_mask` 对应的张量,返回新 dict(原 dict 不动)。
- 末端调用 `core.last_token_only` 把输出收敛成 last-token-only score。

依赖:`torch`、`core.py`。可在 CPU 上以 mock critic 跑测试。

### `critic_update.py` —— Critic update loop

- `update_critic_step(critic_worker, data: DataProto) -> DataProto` —— 替换 `dp_critic.update_critic` 主体。结构:
  1. 切 micro-batch。
  2. 对每个 micro-batch,分别拿 student vpreds(`compute_teacher=False`)和 teacher vpreds(`compute_teacher=True`)。
  3. 调 `core.compute_discriminator_loss` 得 `d_loss`。
  4. 累积梯度 / 动态 bsz scale / `loss.backward()`。
  5. 收集 `d_loss`、`d_acc`、`student_value_mean`、`teacher_value_mean`、`grad_norm` 到 metrics。
- `dp_critic.update_critic` 顶部分支:`if cfg.gad.enable: return update_critic_step(self, data)`,**完全跳过原 verl 实现**。

依赖:`critic_forward.py`、`core.py`、verl 的 `append_to_dict` 与 `DataProto`(import 而非重定义)。

### `rollout_passthrough.py` —— Rollout 透传

- `pass_teacher_response_through(gen_batch, gen_batch_output) -> gen_batch_output` —— rollout 完成后,把 `teacher_response` / `teacher_attention_mask` 从 `gen_batch` 复制到 `gen_batch_output`(因为 rollout 默认只返回 student response)。返回新对象,不原地改 `gen_batch`。

依赖:`verl.protocol.DataProto`(只用其 union/拷贝 API)。

### `data_contract.py` —— 数据契约

- `GAD_BATCH_KEYS: tuple[str, ...] = ("teacher_response", "teacher_attention_mask")`
- `validate_gad_batch(batch) -> None` —— 检查 batch 含上述 key,且形状与 `input_ids`/`attention_mask` 相容(seq-length 维可以不同,batch 维必须一致)。
- `class GADBatchContractError(ValueError): ...` —— 校验失败时抛,错误信息列出缺失 key 与 batch 实际 key,并指向 `docs/algo/gad.md` 数据准备章节。

依赖:无 verl 强耦合(只用 DataProto 的 `.batch.keys()` 等通用 API)。

### `config.py` —— Config dataclass 与开关查询

- `@dataclass class GADConfig:`
  - `enable: bool = False`
  - `discriminator_init_path: str | None = None`
  - `metrics_prefix: str = "gad"`
- `GADConfig.load_from_omegaconf(cfg) -> GADConfig` —— 从 Hydra 节点构造,做完整校验,问题汇总后抛 `GADConfigError`(**列出全部违反的约束,不是只报第一个**)。
- `is_gad_enabled(cfg) -> bool` —— **唯一的开关查询函数**,verl 一侧所有 if 分支都走它。便于未来 grep 定位接入点。
- `class GADConfigError(ValueError): ...`

依赖:`omegaconf`、`dataclasses`。

### `__init__.py` —— 注册入口

```python
from easyopd.registry import register_method

@register_method("gad")
class GADMethod:
    """GAD: Generative Adversarial Distillation (critic-as-discriminator)."""
    verl_modified_files = (
        "verl/trainer/ppo/ray_trainer.py",
        "verl/workers/critic/dp_critic.py",
    )
    verl_changes_summary = (
        "ray_trainer.py: pop teacher_response into gen_batch; pass through after rollout.",
        "dp_critic.py: dispatch to easyopd.methods.gad.critic_update on gad.enable; "
        "last-token-only output and teacher-key remap in _forward_micro_batch.",
    )
    def __init__(self, config):
        ...
    def train(self):
        ...
```

依赖:`easyopd.registry`、本包内 `config`/`core` 等。

### `README.md`

按 EasyOPD README §6.3 模板:方法简介、对 verl 的修改表格、数据约定(必须带 `teacher_response`)、复现步骤、引用论文。

### 模块依赖图(无环)

```
__init__ ──┬→ config
           ├→ core
           ├→ critic_forward → core
           ├→ critic_update  → critic_forward, core
           ├→ rollout_passthrough
           └→ data_contract
```

`core`、`config`、`data_contract` 是叶子节点,纯 Python/torch,**完全 CPU 单测**。

## 数据流(一个训练 step)

```
[dataloader]
   每条样本带:input_ids, attention_mask, position_ids, teacher_response, teacher_attention_mask
        │
        ▼
ray_trainer.fit() L972
   batch = DataProto.from_single_dict(batch_dict)
        │
        ▼ ❶ [EasyOPD:GAD] batch_keys_to_pop 扩展 GAD_BATCH_KEYS
ray_trainer.fit() L975 gen_batch = batch.pop(...)
        │
        ▼
actor_rollout_wg.generate_sequences(gen_batch)   # 标准 verl rollout
   返回 gen_batch_output(只含 student responses)
        │
        ▼ ❷ [EasyOPD:GAD] pass_teacher_response_through(gen_batch, gen_batch_output)
   gen_batch_output 现在也带 teacher_response/teacher_attention_mask
        │
        ▼
batch = batch.repeat(n).union(gen_batch_output)
        │
        ▼
L1086 values = critic_wg.compute_values(batch)
        │  critic_forward.forward_micro_batch(compute_teacher=False)
        │  → 每条样本的 last token 是 D 给出的分数,其余为 0
        ▼
batch.batch["values"] → token_level_scores → token_level_rewards
        │
        ▼
compute_advantage(...)   # 标准 verl,advantage 来自 D 的最后位置分数
        │
        ▼
L1126 critic_wg.update_critic(batch)
   ❸ [EasyOPD:GAD] dp_critic.update_critic 顶部:
        if is_gad_enabled(self.config):
            return update_critic_step(self, data)
   ── student forward + teacher forward + BT loss + backward + metrics
        │
        ▼
L1135 actor_rollout_wg.update_actor(batch)
   ── 标准 PPO actor 更新,无需改动
        │
        ▼
step 结束;metrics 上报(含 d_loss / d_acc / student_value_mean / teacher_value_mean)
```

## verl 接入点(4 处逻辑入口,共 5 处 `[EasyOPD:GAD]` 注释包裹)

### ❶ `verl/trainer/ppo/ray_trainer.py` ~L975 —— gen_batch pop 列表

```python
# ============ [EasyOPD:GAD] Extra batch keys for teacher response ============
from easyopd.methods.gad.config import is_gad_enabled
batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
if is_gad_enabled(self.config):
    from easyopd.methods.gad.data_contract import GAD_BATCH_KEYS
    batch_keys_to_pop.extend(GAD_BATCH_KEYS)
# ============ [EasyOPD:GAD] End ============
```

diff 估计 **~7 行**。

### ❷ `verl/trainer/ppo/ray_trainer.py` ~L1001 —— rollout 后透传 teacher_response

```python
# ============ [EasyOPD:GAD] Pass teacher_response through rollout ============
if is_gad_enabled(self.config):
    from easyopd.methods.gad.rollout_passthrough import pass_teacher_response_through
    gen_batch_output = pass_teacher_response_through(gen_batch, gen_batch_output)
# ============ [EasyOPD:GAD] End ============
```

diff 估计 **~5 行**。

### ❸ `verl/workers/critic/dp_critic.py::DataParallelPPOCritic.update_critic` 顶部

```python
def update_critic(self, data: DataProto):
    # ============ [EasyOPD:GAD] Discriminator-as-critic update ============
    from easyopd.methods.gad.config import is_gad_enabled
    if is_gad_enabled(self.config):
        from easyopd.methods.gad.critic_update import update_critic_step
        return update_critic_step(self, data)
    # ============ [EasyOPD:GAD] End ============
    # ... 原有 verl update_critic 逻辑保持不变
```

diff 估计 **~7 行**。**完全不删 verl 原有代码**。

### ❹ `verl/workers/critic/dp_critic.py::DataParallelPPOCritic._forward_micro_batch` 顶部 + 末尾

```python
def _forward_micro_batch(self, micro_batch, *, compute_teacher: bool = False):
    # ============ [EasyOPD:GAD] Swap input keys when scoring teacher ============
    from easyopd.methods.gad.config import is_gad_enabled
    gad_active = is_gad_enabled(self.config)
    if gad_active and compute_teacher:
        from easyopd.methods.gad.critic_forward import remap_to_teacher
        micro_batch = remap_to_teacher(micro_batch)
    # ============ [EasyOPD:GAD] End ============
    # ... 原有 forward 逻辑 ...
    # ============ [EasyOPD:GAD] Reduce to last-token-only score ============
    if gad_active:
        from easyopd.methods.gad.core import last_token_only
        values = last_token_only(values, response_mask)
    # ============ [EasyOPD:GAD] End ============
    return values
```

diff 估计 **~12 行**(两处包裹)。

**签名细节:** `_forward_micro_batch` 新增 `compute_teacher: bool = False` 关键字参数。默认值 `False` 保证 GAD 禁用路径下所有现有调用方零影响;GAD 启用路径下,`update_critic_step` 是**唯一**显式传 `compute_teacher=True` 的入口。

## EasyOPD Hydra 配置(草案)

`easyopd/config/gad/base.yaml`:

```yaml
defaults:
  - _self_
  - /algorithm: gae          # GAD 用 GAE advantage(critic.compute_values 已给 token_level_rewards)
  - /trainer: ppo

gad:
  enable: true
  discriminator_init_path: ???   # 用户必填的 discriminator checkpoint 路径

critic:
  model:
    path: ${gad.discriminator_init_path}   # 直接复用 verl 现成 critic worker 加载机制
  # 其余字段沿用 verl 默认
```

`examples/gad_trainer/train_gad.sh` 是一段 Hydra 启动:

```bash
python -m verl.trainer.main_ppo --config-name <path-to-gad-yaml> ...
```

## 错误处理

按"早失败、明确信息、不写自动 fallback"。

### 数据契约违反

- `data_contract.validate_gad_batch(batch)` 在 trainer fit 循环开头(❶ if-branch 内、pop 之前)调用一次;失败抛 `GADBatchContractError`。
- 错误信息列出缺哪些 key、batch 实际有哪些 key、指向 `docs/algo/gad.md` 数据准备章节。
- 不在每个 micro-batch 反复校验。

### 配置错误

- `GADConfig.load_from_omegaconf(cfg)` 加载时做完整校验,问题汇总后一次性抛 `GADConfigError`。
- 校验在 `GADMethod.__init__` 触发(进入 fit 之前),不让训练跑 N 步才挂。
- 已知失败场景:
  - `gad.enable=true` 但 `gad.discriminator_init_path` 为 `???`。
  - `gad.enable=true` 但 critic worker 未启用。
  - `gad.enable=true` 同时 `reward_model.enable=true`(GAD 用 critic 当 reward,不能再挂 RM)。

### 运行时分支漂移

- `is_gad_enabled(cfg)` 是开关查询的**唯一**入口,所有 verl 一侧 if 分支都走它;升级 verl 时 grep `[EasyOPD:GAD]` 即可定位所有接入点。
- `_forward_micro_batch` 的 `compute_teacher` 默认 `False`:即使 verl 内部新增调用方忘记传,GAD-禁用路径无影响。
- `test_no_drift.py` grep verl 源码,断言 `[EasyOPD:GAD]` 注释行**恰好 10 行**(4 处逻辑入口 = 5 对开始/End 包裹 = 10 行)。

### 失败信息样例

```
GADBatchContractError: GAD is enabled (cfg.gad.enable=true) but batch is missing
required keys. Missing: ['teacher_response', 'teacher_attention_mask'].
Batch has keys: ['input_ids', 'attention_mask', 'position_ids', 'uid'].
See docs/algo/gad.md §Data preparation for the required schema.
```

```
GADConfigError: GAD config has 2 problems:
  - gad.discriminator_init_path is required when gad.enable=true (got '???')
  - gad.enable=true is incompatible with reward_model.enable=true
    (GAD uses critic as the reward source). Set reward_model.enable=false.
```

### 不做的兜底

- 不在 dataset 缺 `teacher_response` 时静默关 GAD —— 直接抛错。
- 不为 `compute_values` 输出形状不符做兼容 —— 直接抛错。
- 不在 BT loss 出 NaN 时自动 skip step —— 直接抛错。
- 不做 discriminator checkpoint 自动下载 —— 用户必填,文件不存在直接报。

## 测试策略(CPU 合同测试,无 GPU)

延续 ROPD 范式:全部 CPU 可跑,不下载模型权重,不依赖 vllm,不起 ray。

### 测试文件清单

| 文件 | 验证什么 | 关键断言 |
|---|---|---|
| `test_imports.py` | 包整体可 import | `from easyopd.methods import gad` 不抛;触发 `@register_method("gad")` |
| `test_registration.py` | 注册表能查到 | `easyopd.registry.get("gad")` 返回 `GADMethod`;`verl_modified_files` 列出 2 个文件 |
| `test_core_numeric.py` | `core.py` 纯函数 | (a) `compute_discriminator_loss` 与手算吻合;(b) `last_token_only` 末位有效、其余 0;(c) `discriminator_accuracy` 边界 |
| `test_data_contract.py` | 数据契约 | 缺 key 抛 `GADBatchContractError`;形状错抛;合法 batch 不抛 |
| `test_config.py` | `GADConfig` 加载校验 | `enable=false` 宽松;`enable=true` 缺 path 抛;错误信息列出全部问题 |
| `test_rollout_passthrough.py` | rollout 透传 | mock DataProto:gen_batch 带 teacher_response、gen_batch_output 不带;调用后 gen_batch_output 含 teacher_response |
| `test_critic_forward.py` | forward 适配 | mock critic_module,验 `compute_teacher` 切换输入键;输出经 `last_token_only` 处理 |
| `test_critic_update_contract.py` | update loop | mock critic + optimizer,CPU 张量跑一个 micro-batch:student/teacher forward 各调一次、`loss.backward()` 被调、metrics 五键齐全 |
| `test_no_drift.py` | verl 接入点完整 | grep 2 个 verl 文件,`# ============ [EasyOPD:GAD]` 注释**恰好 10 行**(4 处逻辑入口 = 5 对开始/End 包裹) |
| `test_no_actor_changes.py` | actor 不被改 | grep `verl/workers/actor/dp_actor.py`,`[EasyOPD:GAD]` 数量 = 0 |
| `test_config_smoke.py` | YAML 可载 | `OmegaConf.load(...)`,defaults 链不出错 |
| `test_entrypoints.py` | 入口 dry-run | `examples/gad_trainer/train_gad.sh --dry-run` 之类,跑到 hydra 解析完成停 |

### 测试基础设施

- `MockCriticModule`(放 `conftest.py`):返回常量 logits 的 `nn.Module`,有 `v_head`/无 `v_head` 两个变体。
- `build_tiny_batch()` fixture:batch_size=4、seq_len=8 的小 DataProto,带齐 GAD 所需键。
- 不引入 pytest 之外的 mocking 库;`unittest.mock.MagicMock` 够用。

### 跑测命令

```bash
pytest tests/easyopd/gad/ -q
```

预期 12 个 test file、~30+ 测试、普通笔记本 CPU < 30 秒跑完。

### 不测的(显式划定边界)

- 不测梯度数值与上游 verl 完全一致(留给 GPU 集群)。
- 不测 Hydra config 与 verl trainer 的端到端组装(涉及 ray)。
- 不测多 GPU sharded 反向传播(verl 自身保证)。
- 不测与 `vllm_rollout_spmd` 的实际交互(只在 trainer 一侧透传,我们碰不到 rollout 内部)。

## 后续阶段(本设计之外)

设计批准后:

1. 由 `superpowers:writing-plans` skill 生成实施计划 `docs/superpowers/plans/2026-05-27-gad-migration-to-easyopd-plan.md`,把上述结构拆成可逐步执行的任务清单。
2. 实施后按 EasyOPD `README §7` 风格分多次 commit,每次 commit 改一类文件(配置、核心模块、verl 接入、测试、文档)。
3. GPU 实跑由人工 owner 在另一个分支或环境进行,本仓库不承诺端到端跑通。
4. 后续工作(不在本期):SeqKD baseline、warmup 阶段、eval-only rollout、discriminator 自动 warmup 工具、teacher 数据准备脚本。
