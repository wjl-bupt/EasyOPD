# Lightning-OPD 集成到 EasyOPD 的总体设计

## 背景

当前有两个相关仓库：

- 目标仓库：`/mnt/d/Area/DL/projects/research/EasyOPD`
- 源仓库：`/mnt/d/Area/DL/projects/research/Lightning-OPD`

`EasyOPD` 是基于 `verl` 的方法集合仓库，已包含 `easyopd/methods/simple`、`easyopd/methods/simct`、`easyopd/methods/sod`、`easyopd/methods/gad` 等方法级目录，以及 `examples/on_policy_distillation_trainer/`、`examples/sft/`、`tests/easyopd/` 等入口与测试布局。EasyOPD 的扩展惯例是「方法包 + verl 侧最小钩子」：

- 方法实现进入 `easyopd/methods/<name>/`。
- verl 侧改动用 `# [EasyOPD:NAME] ... # [EasyOPD:NAME] End` 包裹，以便后续向 upstream verl rebase。
- 方法包的 `register()` 在导入时把扩展点注册回 verl（advantage estimator、reward manager、distillation loss 等）。
- shell 入口放 `examples/<name>_trainer/`，调用 `verl.trainer.main_ppo` 并通过 Hydra override 装配方法所需的配置。

`Lightning-OPD` 是 NVIDIA Jet AI 团队公开的 offline on-policy distillation 研究实现（arXiv:2604.13010），核心贡献是「离线预计算 teacher log-probabilities + teacher consistency」，把标准 OPD 在训练期对 live teacher 服务的依赖完全消除，在单机 8×H100 上把 4B / 8B / 30B-A3B 学生模型的 OPD 训练成本降低 3.6–4.0×，并把 MoE 30B-A3B 这种标准 OPD OOM 的设定带回到可训练范围。其训练框架基于 `slime`（Megatron + Ray + SGLang），不基于 verl。

本次目标不是把 `Lightning-OPD` 原样嵌入 `EasyOPD`，也不是迁移 `slime` 框架，而是把 Lightning-OPD 的方法贡献（离线 teacher logprob 预计算 + on-policy distillation advantage estimator + teacher consistency 约束）抽出来，作为 EasyOPD 中一个正式、可维护、路径清晰的方法模块，建立在 EasyOPD 现有的 verl + on-policy distillation 体系上。

## 目标

1. 将 Lightning-OPD 作为 EasyOPD 的一等方法迁移到 `easyopd/methods/lightning_opd/`。
2. 把 Lightning-OPD 的算法贡献复刻在 verl 之上，**不引入 `slime` 框架**：
   - 数据侧：把 `data_curation/prepare_lightning_opd.py`（含 Phase 1 tokenize + Phase 2 teacher logprob 预计算）改写为 EasyOPD 风格的离线工具，输出带 `teacher_log_probs` 列的 parquet。
   - 训练侧：在 verl 中新增 `on_policy_distillation` advantage estimator，按 token 取 `adv = log P_teacher − log P_student`。
   - 数据装载侧：让 verl 的 dataloader 能读取 parquet 的 `teacher_log_probs` 列，并在 actor batch 中按 response 长度对齐。
3. 把 Lightning-OPD 的「pipeline 命令链」（Step 0–6）以 shell 入口和 README 的形式收敛到 `examples/lightning_opd_trainer/`，包含 prompt 准备、SFT 数据生成、SFT 训练、rollout 收集、teacher logprob 预计算、Lightning-OPD 训练、Megatron→HF checkpoint 转换。
4. 把 Lightning-OPD 的 teacher consistency 约束做成可检查项：SFT teacher 与 OPD teacher 必须同模型；在 prepare 工具和训练入口的 dry-run 输出中显式打印这两个 tokenizer/模型路径，并在两者不一致时拒绝继续。
5. 用 CPU 级合同测试和 dry-run 入口测试证明迁移后的结构可导入、可配置、可执行到命令生成阶段；不在第一阶段跑端到端 GPU 训练。
6. 严格遵守 EasyOPD「最小侵入 verl」原则：所有 verl 侧改动用 `# [EasyOPD:lightning_opd]` 注释包裹；不覆盖 `pyproject.toml` / `setup.py` / `requirements*.txt` 的主线依赖项。
7. 命名策略：目标侧统一使用 `lightning_opd` 这一 snake_case 名称（包名、配置组、入口目录、环境变量前缀）；不保留源仓库的 `slime` / `is_offline_opd` 等内部 sentinel。

## 非目标

第一阶段集成不做以下事情：

- 不迁移 `slime/` 或 `slime_plugins/` 任何一行 Python 实现；`slime` 框架本身不进入 EasyOPD 的依赖图。
- 不迁移 `train.py`、`configs/lightning_opd/qwen3-*-lightning-opd.py`、`configs/opd/qwen3-*-opd.py` 这种基于 slime 的 entrypoint；目标侧统一用 `verl.trainer.main_ppo` + Hydra override。
- 不迁移 `configs/models/qwen3-*.sh` 这些 Megatron 模型架构 shell 脚本；目标侧依赖 verl 已有的模型加载路径。
- 不迁移 `configs/sft/` 下的 `LlamaFactory` 配置文件本身或 `configs/sft/run_sft.sh`；目标侧使用 EasyOPD 现有的 SFT 入口（verl `fsdp_sft_trainer`），并把论文 §3.2 的 SFT 配方翻译进 `easyopd/config/lightning_opd/sft.yaml`（详见「配置设计」一节）。
- 不迁移 `run_docker.sh` 或源仓库的 docker 流水线；目标侧使用 EasyOPD 现有的 `docker/` 资产。
- 不迁移 `assets/`（论文插图、teaser 等）作为 EasyOPD 仓库内容；如有需要由 `docs/algo/lightning_opd.md` 通过外链引用源 README 中的图。
- 不迁移源仓库的 `data/`、`checkpoints/`、`outputs/`、`logs/`、`wandb/`、`.cache/` 等运行产物或数据集快照。
- 不在第一阶段为标准 OPD 增加新的 baseline 入口；EasyOPD 已有 `examples/on_policy_distillation_trainer/` 覆盖该角色，`configs/opd/qwen3-*-opd.py` 不迁移。
- 不在第一阶段实现 sglang rollout 引擎适配；teacher logprob 预计算工具默认走 vLLM（与 EasyOPD/verl 主线 rollout 引擎一致），保留 sglang 作为后续可选 backend。
- 不在第一阶段删除或修改 EasyOPD 现有 ROPD / SOD / SimCT / Simple / GAD 任何方法的代码或入口。
- 不准备直接向 upstream `verl-project/verl` 发 PR；任何 verl 侧改动均以 EasyOPD 内的 marker 注释保留，未来若要外送 PR 需按 `AGENTS.md` 做 duplicate-work checks 和人工责任说明。

## 总体集成原则

### 1. 方法归方法，入口归入口

Lightning-OPD 的 Python 主实现进入方法包：

```text
easyopd/methods/lightning_opd/
```

面向用户运行的 shell 入口、数据准备工具的 CLI 包装、README 进入执行目录：

```text
examples/lightning_opd_trainer/
```

库代码（advantage estimator、data column adapter、teacher consistency 检查、prepare-data Python 入口）放方法包；shell 编排和命令文档放 `examples/`。

### 2. 算法贡献复刻，而非框架迁移

Lightning-OPD 在 slime 中的算法贡献由三个点构成，分别在 EasyOPD 中找对应的扩展点：

| Lightning-OPD（slime） | EasyOPD（verl） |
|---|---|
| `--advantage-estimator on_policy_distillation` 在 `slime/backends/megatron_utils/loss.py` 中 `adv = teacher_lp − student_lp` | 在 `verl/trainer/ppo/core_algos.py` 注册新 `AdvantageEstimator.ON_POLICY_DISTILLATION`，由方法包 `register()` 时挂上 |
| `slime/rollout/on_policy_distillation.py` 的 `reward_func` / `post_process_rewards` 把 parquet 里的 `teacher_log_probs` 注入 `sample.teacher_log_probs` | 在 verl dataloader 或 ray_trainer 数据准备阶段读取 parquet `teacher_log_probs` 列，并按 response 长度截断到 token 级 tensor，挂到 `batch.batch["teacher_log_probs"]` |
| `data_curation/prepare_lightning_opd.py` Phase 1/2 | `easyopd/methods/lightning_opd/data_curation/prepare.py`（Phase 1 tokenize，Phase 2 vLLM teacher logprob 预计算），CLI 包装在 `examples/lightning_opd_trainer/tools/prepare_data.sh` |

不试图把 slime 的 placement group、weight backuper、KV cache offload 等基础设施搬过来；这些能力 verl 已有自己的实现路径，差异由 EasyOPD 配置文档说明。

### 3. 数据只迁移工具，不迁移产物

源仓库 `data/` 与 `checkpoints/` 是用户运行产物，不是迁移依据。目标侧只迁移：

- `data_curation/prepare_lightning_opd.py` 的算法逻辑（重写到 EasyOPD 风格）。
- `data_curation/merge.py` 的等价逻辑（如果 EasyOPD 已有等价 merger 则复用）。
- `data_curation/pipeline.py` 中与 vLLM SFT 数据生成有关的最小逻辑（如果 EasyOPD `examples/data_preprocess/` 已能覆盖则不重复迁移）。
- `scripts/prepare_sft_prompts.py` 的等价 Python 工具（HF dataset → JSONL）。

默认数据路径通过 `LIGHTNING_OPD_*` 环境变量或入口脚本参数指定。仓库里只放工具与 README，不放任何 parquet / arrow / checkpoint。

### 4. 直接替换优先，不主张兼容 alias

迁移时优先把源仓库的命名（`slime.rollout.on_policy_distillation.reward_func`、`is_offline_opd`、`is_lightning_opd`、`teacher_log_probs` 元数据键）直接改写为 EasyOPD 主线路径与命名。不提供 `slime.*` 路径的 shim 模块，不在 verl 中保留 sentinel `is_offline_opd`；目标侧 parquet 的 schema 由 EasyOPD 文档定义，名字使用 `teacher_log_probs`（列名），并以一段 doc 明确该列含义。

只有在以下情况，才考虑非常窄的临时兼容层：发现确有正在运行的外部历史作业仍依赖源仓库命名，且直接替换会导致用户工作流不可接受地断裂。该兼容层不作为默认方案，不作为目标结构的一部分。

### 5. 最小侵入 verl

所有 verl 侧改动遵循 EasyOPD 现有惯例：

- 用 `# [EasyOPD:lightning_opd] ... # [EasyOPD:lightning_opd] End` 注释 marker 包裹。
- 优先在方法包 `register()` 时调用 `register_adv_est("on_policy_distillation")`，避免直接编辑 `core_algos.py` 中的 `AdvantageEstimator` 枚举。
- 如果必须修改 dataloader / trainer 以读取 `teacher_log_probs` 列，把改动限制在 1–2 处明确标注的钩子；其余逻辑在方法包内闭环。

## 目标目录结构

集成完成后，建议新增或调整为：

```text
easyopd/
  config/
    lightning_opd/
      base.yaml
      data_prep.yaml
      training.yaml
      sft.yaml
  methods/
    lightning_opd/
      __init__.py
      method.py
      advantage_estimator.py
      data_adapter.py
      teacher_consistency.py
      data_curation/
        __init__.py
        prepare.py
        merge.py
        prompt_prep.py

examples/
  lightning_opd_trainer/
    README.md
    train_lightning_opd.sh
    tools/
      prepare_sft_prompts.sh
      generate_sft_data.sh
      run_sft.sh
      collect_rollouts.sh
      prepare_data.sh
      convert_megatron_to_hf.sh

docs/
  algo/
    lightning_opd.md
  superpowers/
    specs/
      2026-05-27-lightning-opd-integration-to-easyopd-design.md

tests/
  easyopd/
    lightning_opd/
      __init__.py
      test_method.py
      test_advantage_estimator.py
      test_data_adapter.py
      test_teacher_consistency.py
      test_prepare_pipeline.py
      test_sft_config.py
      test_config_smoke.py
      test_entrypoints.py
      test_no_legacy_names.py
```

`__init__.py` 暴露的公共 surface（与后文代码块中的实际命名保持一致）：

```python
from easyopd.methods.lightning_opd import (
    METHOD,                                       # LightningOPDMethod dataclass，描述 verl 钩子
    register,                                     # 触发 advantage estimator + data adapter 注册
    compute_on_policy_distillation_advantages,    # advantage_estimator.py 中注册到 verl 的函数
    attach_teacher_log_probs,                     # data_adapter.py 中把 parquet 列挂到 batch 的函数
    check_teacher_consistency,                    # teacher_consistency.py 的纯函数检查器
    LightningOPDTeacherInconsistency,             # 自定义异常
    LightningOPDLogprobLengthMismatch,            # 自定义异常
    LightningOPDMissingTeacherLogprobs,           # 自定义异常
)
```

## 模块迁移映射

| 源路径 | 目标路径 | 策略 |
|---|---|---|
| `slime/rollout/on_policy_distillation.py::reward_func` | `easyopd/methods/lightning_opd/data_adapter.py` | 重写为 verl batch adapter；不保留 sglang rm_url 分支（在 EasyOPD 中走 verl rollout，不依赖 sglang sentinel） |
| `slime/rollout/on_policy_distillation.py::post_process_rewards` | `easyopd/methods/lightning_opd/data_adapter.py` | 重写为 verl dataloader 钩子，把 parquet `teacher_log_probs` 列截断到 response_length 并放入 batch |
| `slime/backends/megatron_utils/loss.py::on_policy_distillation` 分支 | `easyopd/methods/lightning_opd/advantage_estimator.py` | 通过 `register_adv_est` 注册 `on_policy_distillation`，输入 `student_log_probs`、`teacher_log_probs`、`response_lengths`，返回每 token 的 `adv = teacher_lp − student_lp` |
| `data_curation/prepare_lightning_opd.py` Phase 1 | `easyopd/methods/lightning_opd/data_curation/prepare.py::phase1_tokenize` | 直接迁移：tokenizer apply_chat_template + 截断到 `max_response_len`；移除 sglang 专属字段 |
| `data_curation/prepare_lightning_opd.py` Phase 2 | `easyopd/methods/lightning_opd/data_curation/prepare.py::phase2_logprobs` | 改写为基于 vLLM 的 teacher logprob 预计算（与 EasyOPD 现有 vLLM 基础设施一致）；保留 sglang 作为可选 backend 但不在第一阶段实现 |
| `data_curation/merge.py` | `easyopd/methods/lightning_opd/data_curation/merge.py`（或复用 `easyopd` 现有 merger） | 如果 EasyOPD 已有等价 arrow→parquet merger 则不重复迁移，文档说明使用现有工具的命令 |
| `scripts/prepare_sft_prompts.py` | `easyopd/methods/lightning_opd/data_curation/prompt_prep.py` | 直接迁移；HF dataset → JSONL，支持 `--input-parquet` 本地源 |
| `scripts/generate_sft_data.sh` | `examples/lightning_opd_trainer/tools/generate_sft_data.sh` | 改写为调用 EasyOPD `examples/data_preprocess/` 或 verl rollout 工具 |
| `scripts/collect_rollouts.sh` | `examples/lightning_opd_trainer/tools/collect_rollouts.sh` | 同上，使用 EasyOPD 现有 rollout 工具 |
| `scripts/serve_teacher_{8b,32b}.sh` | `examples/lightning_opd_trainer/tools/serve_teacher.sh` | 改写为 vLLM teacher 服务启动脚本；URL 通过 `LIGHTNING_OPD_TEACHER_URL` 环境变量传递；sglang 版本不在第一阶段迁移 |
| `scripts/precompute_teacher_logprobs_{4b,8b}.sh` | `examples/lightning_opd_trainer/tools/prepare_data.sh` | 单一入口完成 Phase 1 + Phase 2，参数化模型规模 |
| `scripts/convert_megatron_to_hf.sh` | `examples/lightning_opd_trainer/tools/convert_megatron_to_hf.sh` | 直接迁移并去掉 slime 路径假设；如果 verl 已有等价 converter 则复用 |
| `configs/lightning_opd/qwen3-4b-lightning-opd.py` | `easyopd/config/lightning_opd/training.yaml` + `examples/lightning_opd_trainer/train_lightning_opd.sh` 中的 4B override 段 | 翻译为 verl Hydra override；slime 专属 flag（`--use-dynamic-batch-size`、`--sglang-mem-fraction-static`、`--rollout-num-gpus`）由 verl 等价配置替代 |
| `configs/lightning_opd/qwen3-8b-lightning-opd.py` | 同上 8B override 段 | 同上 |
| `configs/lightning_opd/qwen3-30b-a3b-lightning-opd.py` | 同上 30B-A3B override 段 | 同上；MoE 的 EP/TP 配置改用 verl Megatron worker 的等价 flag |
| `configs/sft/*.yaml`（LlamaFactory 配置） | `easyopd/config/lightning_opd/sft.yaml`（verl SFT 键名） | 不照搬 LlamaFactory 文件；按论文 §3.2 把 3000-step / lr=8e-5 / packing-enabled 等关键字段翻译进 verl `fsdp_sft_trainer` 配置；不一致字段在 sft.yaml 中加注释解释 |
| `configs/sft/run_sft.sh` | `examples/lightning_opd_trainer/tools/run_sft.sh` | thin wrapper 调用 verl `fsdp_sft_trainer` + `easyopd/config/lightning_opd/sft.yaml`；与 `train_lightning_opd.sh` 同样的 `LIGHTNING_OPD_*` 环境变量与 dry-run 约定 |
| `configs/models/qwen3-*.sh` | 不迁移；模型架构由 verl model loader 决定 | 文档化 |
| `train.py` | 不迁移 | EasyOPD 主入口是 `verl.trainer.main_ppo` |
| `assets/*.png` | 不迁移；`docs/algo/lightning_opd.md` 外链源 README | 文档化 |

## 核心算法集成设计

Lightning-OPD 的算法本质可以拆为三块。EasyOPD 的实现要让这三块各自有清晰的扩展点。

### A. on-policy distillation advantage estimator

注册函数挂在 `easyopd/methods/lightning_opd/advantage_estimator.py`：

```python
# easyopd/methods/lightning_opd/advantage_estimator.py
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("on_policy_distillation")
def compute_on_policy_distillation_advantages(
    *,
    student_log_probs,           # list[Tensor[response_len_i]] or Tensor[B, T]
    teacher_log_probs,           # list[Tensor[response_len_i]]
    response_lengths,            # list[int]
    **kwargs,                    # 兼容 verl 现有 adv 估计器的多余参数
):
    """Per-token advantage = teacher_log_prob - student_log_prob.

    Lightning-OPD 论文 §3: this signal is the exact KL gradient surrogate
    for the distillation objective, given the offline teacher consistency
    condition. Returns advantages with the same shape/length as
    `student_log_probs`.
    """
    ...
```

注册时机由 `easyopd/methods/lightning_opd/__init__.py::register()` 触发：

```python
def register() -> None:
    from .advantage_estimator import compute_on_policy_distillation_advantages  # noqa: F401
    from .data_adapter import register_data_adapter
    register_data_adapter()
```

入口脚本通过 `algorithm.adv_estimator=on_policy_distillation` 选用。

### B. teacher_log_probs 数据列适配

`easyopd/methods/lightning_opd/data_adapter.py` 负责：

- 在 dataloader/ray_trainer 侧识别 batch 中存在的 `teacher_log_probs` 列（list 或 ragged tensor）。
- 把每条样本的 teacher logprob 序列截断到 `response_length`（与 student_log_probs 对齐）。
- 注入到 actor batch 的 `batch.batch["teacher_log_probs"]`，让 advantage estimator 能拿到。

实现优先级（按侵入性递增）：

1. **首选**：在 verl 现有的 dataloader / batch builder 中加一个轻量 hook（marker 包裹），按列名约定 `teacher_log_probs` 自动透传。该 hook 用 `getattr(batch, "teacher_log_probs", None)`，对未使用的 batch 完全无副作用。
2. **次选**：在方法包内实现一个 `RayTrainer` 子类或 `Batch` postprocessor，由入口脚本通过 `+trainer.batch_postprocessor=easyopd.methods.lightning_opd.data_adapter:attach_teacher_log_probs` 接入。

第一阶段实现选「首选」，但所有 verl 侧改动必须在 marker 内并控制在 ≤ 10 行。

### C. teacher consistency 检查

`easyopd/methods/lightning_opd/teacher_consistency.py` 提供一个纯函数级检查器：

- 输入：SFT 步骤使用的 teacher 标识符、Lightning-OPD prepare 步骤使用的 teacher 标识符（HF path 或 sha256 of tokenizer.json）。
- 输出：一致返回 `True`，否则抛 `LightningOPDTeacherInconsistency` 异常。

接入位置：

- prepare 工具（`prepare.py`）的入口：当传入 `--sft-teacher-id` 与 `--opd-teacher-id` 且不一致时拒绝继续；可以用 `--allow-teacher-mismatch` 显式覆盖（仅用于调试，需要 wandb 上独立 tag）。
- 训练入口的 dry-run：打印两个 teacher 路径并显示一致/不一致状态。

该检查不在 verl 内强制启用，但在 EasyOPD 文档和 dry-run 输出中作为「Lightning-OPD 必读约束」明确列出。

## 配置设计

新增配置目录：

```text
easyopd/config/lightning_opd/
```

建议分层：

- `base.yaml`：Lightning-OPD 训练顶层默认。
  - `algorithm.adv_estimator: on_policy_distillation`
  - `algorithm.use_kl_in_reward: False`
  - `distillation.enabled: False`（不启用 verl 的 live teacher Ray cluster）
  - `actor_rollout_ref.rollout.n: 1`（与论文一致）
  - `actor_rollout_ref.actor.optim.lr: 2e-6` constant
- `training.yaml`：训练超参（global batch、max response length、TP/EP 等）的默认模板。可被 4B/8B/30B-A3B override 替换。
- `data_prep.yaml`：数据准备阶段（prepare、prompt_prep）的默认参数。
- `sft.yaml`：Lightning-OPD SFT warmup 的 verl SFT 键名配方，翻译自论文 §3.2 与源仓库 `configs/sft/qwen3-4b-base-open-thoughts3-qwen3-8b.yaml`：
  - `trainer.total_training_steps: 3000`
  - `optim.lr: 8.0e-5`
  - `data.use_dynamic_bsz: True`、`data.train_batch_size: 256`、`data.max_length: 32768`
  - `model.enable_packing: True`（或 verl 等价字段；具体键名在实现时按 verl `fsdp_sft_trainer` 当前 schema 校准）
  - `model.path: <SFT base model>`
  - 头注释明确「该配置对应论文 §3.2 / 源仓库 LlamaFactory 配置的 verl 翻译；任何关键字段（total_training_steps / lr / packing）的修改需要走 spec 评审，因为这是 Lightning-OPD 复现 reported numbers 的必要条件」。
  - LlamaFactory 与 verl 之间无法 1:1 对齐的字段（如 LoRA 设置、deepspeed config 等）在 sft.yaml 中以注释列出并说明 EasyOPD 侧的等价做法。

入口脚本通过 Hydra override 组合：

```bash
python3 -m verl.trainer.main_ppo \
    --config-path "$PROJECT_ROOT/easyopd/config/lightning_opd" \
    --config-name base \
    algorithm.adv_estimator=on_policy_distillation \
    data.train_files=... \
    ...
```

配置命名策略：

- 新配置只使用 `lightning_opd`；不引入 `slime_*`、`offline_opd` 等命名。
- 环境变量前缀统一 `LIGHTNING_OPD_*`：
  - `LIGHTNING_OPD_PROJECT_ROOT`、`LIGHTNING_OPD_DRYRUN`、`LIGHTNING_OPD_SKIP_REPO_DOTENV`
  - `LIGHTNING_OPD_SFT_CHECKPOINT`、`LIGHTNING_OPD_TEACHER_MODEL`、`LIGHTNING_OPD_TEACHER_URL`
  - `LIGHTNING_OPD_DATA`（precomputed parquet 路径）
  - `LIGHTNING_OPD_PROMPTS`（DAPO-Math 等 OPD prompt 数据）

不引入 `SFT_CHECKPOINT` / `OPD_PROMPTS` 这种短前缀全局变量（与源仓库不同），避免和 EasyOPD 其他 trainer 的环境变量冲突。

## 执行入口设计

唯一保留的一线 shell 训练命令集中在：

```text
examples/lightning_opd_trainer/
```

主入口：

```bash
bash examples/lightning_opd_trainer/train_lightning_opd.sh
```

入口脚本约束（与 ROPD `train_ropd.sh` 一致）：

- 从 `BASH_SOURCE[0]` 解析 `PROJECT_ROOT`，可从任意 cwd 调用。
- 支持 `LIGHTNING_OPD_DRYRUN=true`：组装命令但不执行，打印完整 Hydra 命令、`LIGHTNING_OPD_DATA`、SFT/teacher checkpoint 路径、teacher consistency 检查结果。
- 支持 `LIGHTNING_OPD_SKIP_REPO_DOTENV=true`：避免读本地密钥（用于 CI）。
- 支持 `LIGHTNING_OPD_PROJECT_ROOT` 覆盖路径解析（用于测试）。
- 通过位置参数或环境变量选模型规模：`MODEL_SCALE=4b|8b|30b-a3b`，把对应规模的 TP/EP/batch 默认值套用进 Hydra override。
- Python 逻辑下沉到 `easyopd.methods.lightning_opd`；shell 只保留命令编排。

数据准备入口集中在 `examples/lightning_opd_trainer/tools/`：

- `prepare_sft_prompts.sh`：HF dataset → JSONL（Step 0）。
- `generate_sft_data.sh`：teacher 生成 SFT 训练数据（Step 1）。
- `run_sft.sh`：调用 verl `fsdp_sft_trainer` + `easyopd/config/lightning_opd/sft.yaml`，完成论文 §3.2 配方的 SFT 训练（Step 2）。
- `collect_rollouts.sh`：SFT 学生 rollout（Step 3）。
- `prepare_data.sh`：Phase 1 tokenize + Phase 2 teacher logprob 预计算（Step 4）。
- `convert_megatron_to_hf.sh`：Megatron → HF（Step 6）。

每个 tool 脚本都支持 dry-run。Step 5 的训练入口由根目录的 `train_lightning_opd.sh` 承担；`sft.yaml` 的关键字段被 `test_sft_config.py` 守护，任何对 `total_training_steps` / `lr` / `packing` 的修改需要走 spec 评审。

## 文档设计

新增：

```text
docs/algo/lightning_opd.md
examples/lightning_opd_trainer/README.md
```

`docs/algo/lightning_opd.md` 说明：

- Lightning-OPD 方法定义、与标准 OPD 的差异、引用论文（arXiv:2604.13010）。
- Teacher consistency 约束作为「先决条件」单独 callout，强调 SFT teacher 必须等于 OPD teacher。
- 完整 7 步 pipeline 表（Step 0–6），列出每步的 EasyOPD 入口、关键环境变量、产物路径。
- Parquet schema 约定（列名、类型、含义）。
- `algorithm.adv_estimator=on_policy_distillation` 的使用方式与对应的 batch 列要求。
- 关键配置与环境变量列表（`LIGHTNING_OPD_*`、`MODEL_SCALE`）。
- 与论文结果的差异说明（默认 vLLM backend；30B-A3B 第一阶段不端到端验证）。
- 与 EasyOPD 现有 `examples/on_policy_distillation_trainer/` 的关系（live teacher vs 离线 logprob 路径并存）。

`examples/lightning_opd_trainer/README.md` 说明：

- 最小 dry-run 命令（训练入口 + prepare 入口）。
- 完整 7 步运行命令链（Step 0–6），按论文 §A 顺序排列。
- 推荐 `LIGHTNING_OPD_DATA` 与 `LIGHTNING_OPD_SFT_CHECKPOINT` 目录布局。
- 常用环境变量与默认值。
- 不迁移内容的清单与原因（slime、LlamaFactory configs、源仓库 docker 等）。
- 4B / 8B 路径已验证范围；30B-A3B 标记为 stretch。

`docs/index.rst` 在 algo toctree 中加入 `algo/lightning_opd`，与现有 `algo/ropd`、`algo/sod` 等并列。

`docs/superpowers/specs/` 保存本设计文档（即本文件），用于后续写 implementation plan。

## 数据流水线设计

Lightning-OPD 完整 pipeline 是 7 步（Step 0–6）。目标侧在 README 中以表格形式给出每一步的 EasyOPD 对应入口、所需环境变量、推荐硬件与产物路径：

| Step | 描述 | EasyOPD 入口 | 关键环境变量 |
|---|---|---|---|
| 0 | 准备 SFT prompts | `examples/lightning_opd_trainer/tools/prepare_sft_prompts.sh` | `LIGHTNING_OPD_HF_DATASET`, `LIGHTNING_OPD_OUT` |
| 1 | 生成 SFT 训练数据 | `examples/lightning_opd_trainer/tools/generate_sft_data.sh` | `LIGHTNING_OPD_TEACHER_MODEL`, `LIGHTNING_OPD_SFT_PROMPTS`, `LIGHTNING_OPD_OUT` |
| 2 | SFT 训练 | `examples/lightning_opd_trainer/tools/run_sft.sh`（thin wrapper，调用 verl `fsdp_sft_trainer` + `easyopd/config/lightning_opd/sft.yaml`） | `LIGHTNING_OPD_SFT_BASE_MODEL`, `LIGHTNING_OPD_SFT_DATA`, `LIGHTNING_OPD_SFT_OUT` |
| 3 | 学生 rollout | `examples/lightning_opd_trainer/tools/collect_rollouts.sh` | `LIGHTNING_OPD_SFT_CHECKPOINT`, `LIGHTNING_OPD_OPD_PROMPTS`, `LIGHTNING_OPD_OUT` |
| 4 | Phase 1 tokenize + Phase 2 teacher logprob 预计算 | `examples/lightning_opd_trainer/tools/prepare_data.sh` | `LIGHTNING_OPD_TOKENIZER`, `LIGHTNING_OPD_ROLLOUTS`, `LIGHTNING_OPD_TEACHER_URL`, `LIGHTNING_OPD_OUT` |
| 5 | Lightning-OPD 训练 | `examples/lightning_opd_trainer/train_lightning_opd.sh` | `LIGHTNING_OPD_SFT_CHECKPOINT`, `LIGHTNING_OPD_DATA`, `MODEL_SCALE` |
| 6 | Megatron → HF | `examples/lightning_opd_trainer/tools/convert_megatron_to_hf.sh` | `MEGATRON_CKPT_DIR`, `HF_OUTPUT_DIR`, `ORIGIN_HF_DIR` |

第一阶段优先把 Step 4、Step 5 的 EasyOPD 入口实现并测通；Step 0、1、3、6 改写为 thin shell wrapper 调用 EasyOPD 现有工具，但仍归属 `examples/lightning_opd_trainer/tools/` 以便用户从一个目录看完整链路。

Parquet schema 约定（输出的 Lightning-OPD 训练数据）：

```text
列名                类型           描述
prompt              str            apply_chat_template 输出
label               str            可选；下游评估用
response_tokens     list[int]      学生 rollout 的 response token ids
response_length     int            response token 数（截断到 max_response_len）
teacher_log_probs   list[float]    长度 == response_length；按 response token 顺序排列
metadata            dict           可选附加信息（sft_teacher_id、opd_teacher_id、tokenizer hash）
```

`metadata` 中存的 `sft_teacher_id` 与 `opd_teacher_id` 用于 teacher consistency 检查；不存大对象。

## 测试与验收

### 静态和导入检查

集成后应通过：

```bash
python -m compileall easyopd/methods/lightning_opd
```

并确保不存在源仓库路径残留：

```bash
rg -n "from slime|import slime|slime\.rollout|slime\.backends|slime_plugins" easyopd/methods/lightning_opd examples/lightning_opd_trainer tests/easyopd/lightning_opd
```

预期：无匹配。允许 `tests/easyopd/lightning_opd/test_no_legacy_names.py` 中出现作为反例的字符串字面量。

### 单元测试

```bash
uv run --no-sync pytest tests/easyopd/lightning_opd -v
```

第一阶段至少覆盖：

- `test_advantage_estimator.py`：
  - `compute_on_policy_distillation_advantages` 返回 `teacher_lp − student_lp`。
  - 处理 ragged batch（不同 response_length）。
  - 处理空 teacher_log_probs（应抛 `LightningOPDMissingTeacherLogprobs`）。
  - 注册名 `on_policy_distillation` 在 `register_adv_est` 后可被 `get_adv_estimator_fn("on_policy_distillation")` 拿到。
- `test_data_adapter.py`：
  - Parquet `teacher_log_probs` 列按 `response_length` 截断后挂到 batch。
  - response_length 与 teacher_log_probs 长度不一致时抛 `LightningOPDLogprobLengthMismatch`。
  - batch 不含该列时 adapter 为 no-op，不改动 batch。
- `test_teacher_consistency.py`：
  - SFT teacher id == OPD teacher id 时返回 `True`。
  - id 不一致时抛 `LightningOPDTeacherInconsistency`。
  - `--allow-teacher-mismatch` 时降级为 warning，不抛。
- `test_prepare_pipeline.py`：
  - Phase 1 tokenize：构造小规模 fake parquet，确认 response 截断到 `max_response_len`、`response_length` 列存在、`teacher_log_probs` 列未写入。
  - Phase 2（mock teacher URL）：Phase 1 输出 + mocked aiohttp 返回 logprob 序列 → 写入 `teacher_log_probs` 列；缺 response 时 skip 而非崩。
  - 端到端 (Phase 1 + Phase 2 mock) 后的 parquet 满足上述 schema。
- `test_method.py`：
  - `METHOD.name == "lightning_opd"`。
  - `register()` 调用后 `get_adv_estimator_fn("on_policy_distillation")` 可解析。
  - `register()` idempotent。
- `test_config_smoke.py`：
  - `easyopd/config/lightning_opd/base.yaml` 能被 OmegaConf 解析。
  - `base.yaml` 中 `algorithm.adv_estimator == "on_policy_distillation"`。
  - `data_prep.yaml` 暴露 `max_response_len`、`concurrency` 等键。
  - `sft.yaml` 能被 OmegaConf 解析（结构合法性；语义断言放 `test_sft_config.py`）。
- `test_sft_config.py`（论文 §3.2 配方合同测试）：
  - `easyopd/config/lightning_opd/sft.yaml` 中 `trainer.total_training_steps == 3000`。
  - `optim.lr == 8.0e-5`（允许 `8e-5` 表达，断言数值相等）。
  - `data.train_batch_size == 256`。
  - `data.use_dynamic_bsz` 与 packing 相关键存在且为 True。
  - 头注释中包含「§3.2」字符串，用于人工追溯出处。
  - 任何上述字段缺失或被改动都让该测试失败 → 强迫修改者走 spec 评审。
- `test_entrypoints.py`：
  - `LIGHTNING_OPD_DRYRUN=true LIGHTNING_OPD_SKIP_REPO_DOTENV=true bash examples/lightning_opd_trainer/train_lightning_opd.sh` 退出 0，stdout 包含 `PROJECT_ROOT`、`MODEL_SCALE`、`LIGHTNING_OPD_DATA`、最终 python 命令、`adv_estimator=on_policy_distillation`。
  - `tools/prepare_data.sh` dry-run 退出 0，打印 Phase 1/Phase 2 命令、teacher consistency 状态。
  - `LIGHTNING_OPD_PROJECT_ROOT` override 被尊重。
- `test_no_legacy_names.py`：
  - 扫描 `easyopd/methods/lightning_opd`、`easyopd/config/lightning_opd`、`examples/lightning_opd_trainer`、`docs/algo/lightning_opd.md`，断言没有 `slime\.`、`SLIME_`、`is_offline_opd`、`is_lightning_opd`（这是 slime 内部 sentinel）、`LightningOpd`（错误大小写）、`lightning-opd`（应该是 `lightning_opd`）等 token 出现。

### verl 钩子合同测试

确认 verl 侧改动符合「marker 包裹」+「方法包加载触发注册」：

- `rg -n "EasyOPD:lightning_opd" verl/` 至少匹配一次。
- 不导入 `easyopd.methods.lightning_opd` 时，`get_adv_estimator_fn("on_policy_distillation")` 抛 `KeyError`；导入后能解析。

### 入口 dry-run

至少通过：

```bash
LIGHTNING_OPD_DRYRUN=true LIGHTNING_OPD_SKIP_REPO_DOTENV=true \
    bash examples/lightning_opd_trainer/train_lightning_opd.sh
```

dry-run 必须打印：

- 解析后的 `PROJECT_ROOT`
- `MODEL_SCALE` 与对应模板路径
- `LIGHTNING_OPD_DATA`、`LIGHTNING_OPD_SFT_CHECKPOINT`
- teacher consistency 检查结果
- 完整 `python3 -m verl.trainer.main_ppo ...` 命令，含 `algorithm.adv_estimator=on_policy_distillation`

第一阶段不要求端到端 GPU 训练通过；GPU 训练放到 Phase 5 验收（见「实施切分建议」）。

## 风险与缓解

### 风险 1：slime ↔ verl 数据流抽象不同，advantage estimator 拿不到 teacher_log_probs

slime 的 `rollout_data["teacher_log_probs"]` 是在 rollout postprocess 中显式挂到 sample 上的 list；verl 的 actor batch 是 `DataProto`，没有现成的 ragged tensor 列约定。

缓解：

- 在 `data_adapter.py` 中显式实现 list[Tensor] 与 `DataProto` 之间的双向桥接，单元测试覆盖 ragged batch。
- 不让 advantage estimator 直接读 dataloader；让它只接受 `student_log_probs`、`teacher_log_probs`、`response_lengths` 三个参数，从而把 verl 内部数据结构隔离在 adapter 一层。
- 在 dry-run 命令中打印「teacher_log_probs column found in: ...」以便用户尽早发现 schema 不对。

### 风险 2：teacher consistency 没有强制检查，用户用错 teacher 会得到错误结论

论文 §3 指出 SFT teacher ≠ OPD teacher 时存在不可消除的 gradient bias。若 EasyOPD 不在 prepare/training 入口提示，用户很可能踩坑。

缓解：

- `teacher_consistency.py` 在 prepare 工具中 **默认开启** 检查，不一致时直接退出非 0；只有显式 `--allow-teacher-mismatch` 才降级 warning。
- `docs/algo/lightning_opd.md` 在「先决条件」一节用单独 callout 强调 teacher consistency 必须。
- `test_teacher_consistency.py` 覆盖默认 strict 行为 + opt-in mismatch 行为。

### 风险 3：vLLM 与 sglang 的 logprob 数值精度不一致

源仓库用 sglang 的 `/generate` + `return_logprob` 取 token logprob。EasyOPD 主线 rollout 用 vLLM。两者在 fp16/bf16 数值精度、tokenizer 特殊 token 处理上可能有微小差异，导致复现论文数字时偏移。

缓解：

- 第一阶段在 `docs/algo/lightning_opd.md` 中明确：默认 vLLM backend，与论文结果**不要求**完全一致；如需精确复现，参考源仓库 sglang 路径。
- `prepare_data.sh` 输出 metadata 中记录 backend 名称、版本、关键 sampling params，便于事后归因。
- 在 `test_prepare_pipeline.py` 中 mock 而不是真的依赖 vLLM；端到端数值复现作为后续 GPU 验收任务。

### 风险 4：slime 命名残留导致 verl/EasyOPD 公共 surface 被污染

源仓库使用 `is_offline_opd` / `is_lightning_opd` / `slime.rollout.on_policy_distillation` 等命名作为内部 sentinel。直接迁移容易污染目标公共 surface。

缓解：

- 目标侧严禁出现 `slime.*`、`is_offline_opd`、`is_lightning_opd`（除非作为反例字符串）。
- `test_no_legacy_names.py` 在 CI 跑，作为合同测试守护。
- adv_estimator 的注册名是 `on_policy_distillation`（与论文术语对齐，并使用 EasyOPD 标准 snake_case），不沿用 slime 的 `--advantage-estimator` 命名歧义。

### 风险 5：MoE（Qwen3-30B-A3B）配置无法在 verl 中直接落地

slime 的 30B-A3B 配置使用 Megatron 的 EP=8 / TP=4 + `--use-dynamic-batch-size`。verl 的 Megatron worker 等价配置需要单独验证。

缓解：

- 第一阶段 MoE 配置作为 **stretch 目标**：在 `easyopd/config/lightning_opd/training.yaml` 中提供 30B-A3B 默认模板，但 entrypoint dry-run 只测 4B 路径。
- `docs/algo/lightning_opd.md` 明确：30B-A3B 配置尚未在 EasyOPD 内端到端验证，先交付 4B / 8B 路径。
- 后续单独 spec 跟进 verl Megatron MoE 与 slime Megatron MoE 的等价性。

### 风险 6：verl 侧改动越界

为了让 dataloader 支持 `teacher_log_probs` 列，可能需要触碰 verl 中 1–2 处文件。如果一开始没有边界感，容易越改越多。

缓解：

- 在写 implementation plan 时把 verl 侧改动总行数硬性上限定在 **≤ 20 行**，超出需要回到 spec 阶段重新设计。
- 所有改动用 `# [EasyOPD:lightning_opd]` marker 包裹，方便后续 audit。
- 在 `test_method.py` 中加一条「marker grep 必须命中」的合同测试。

### 风险 7：数据准备工具的 vLLM 依赖污染核心依赖

`prepare_data.sh` 调用 vLLM 启 teacher 服务；如果把 vLLM 升到核心 requirements，可能与 EasyOPD 主线版本冲突。

缓解：

- 第一阶段不修改 `pyproject.toml` / `requirements*.txt`；prepare 工具假设 vLLM 已由 EasyOPD 主线安装。
- 如确有版本冲突，把 Lightning-OPD 数据准备工具的额外依赖放 optional extra `lightning_opd_prep`，而不是改主依赖。

## 实施切分建议

### Phase 1：方法包骨架 + advantage estimator

- 新建 `easyopd/methods/lightning_opd/{__init__.py, method.py, advantage_estimator.py}`。
- 注册 `on_policy_distillation` advantage estimator。
- 新建 `tests/easyopd/lightning_opd/{test_method.py, test_advantage_estimator.py, test_no_legacy_names.py}` 并通过。

验收：方法包可 import；`get_adv_estimator_fn("on_policy_distillation")` 可解析；CPU 单测通过。

### Phase 2：数据列 adapter + teacher consistency

- 实现 `data_adapter.py` 把 parquet `teacher_log_probs` 列接入 actor batch。
- 实现 `teacher_consistency.py` 检查器。
- 单测覆盖 ragged batch、length mismatch、teacher mismatch 三类异常。
- 必要时在 verl dataloader 中加一处 marker hook（≤ 10 行）。

验收：合成 parquet + mock batch 走通 adapter；teacher consistency 合同通过。

### Phase 3：数据准备工具

- `data_curation/prepare.py` Phase 1 + Phase 2 入口。
- `data_curation/prompt_prep.py`（HF → JSONL）。
- `examples/lightning_opd_trainer/tools/prepare_data.sh` 和 `prepare_sft_prompts.sh`。
- 合成 parquet + mock teacher URL 的端到端单测。

验收：mock 模式下 prepare 完整链路通过；产出的 parquet schema 满足约定。

### Phase 4：训练入口 + 配置 + 文档

- `easyopd/config/lightning_opd/{base.yaml, training.yaml, data_prep.yaml, sft.yaml}`。
- `examples/lightning_opd_trainer/train_lightning_opd.sh`（dry-run 支持）。
- `examples/lightning_opd_trainer/tools/{prepare_sft_prompts.sh, generate_sft_data.sh, run_sft.sh, collect_rollouts.sh, convert_megatron_to_hf.sh}`。
- `docs/algo/lightning_opd.md` + `docs/index.rst` toctree entry。
- `examples/lightning_opd_trainer/README.md`。
- `test_config_smoke.py`、`test_sft_config.py`、`test_entrypoints.py`。

验收：dry-run 入口测试通过；`docs/algo/lightning_opd.md` 渲染无错；`git status` 只包含预期文件。

### Phase 5（后续，可选）：GPU 端到端验证

- 4B 规模端到端跑 100 step，对比论文 reward 曲线。
- 8B 规模冒烟 1 step。
- 30B-A3B 配置在 verl 上独立验证（可能需要单独 spec）。

不在第一阶段验收范围内，但在 spec 中记录为后续路线。

## 开放问题

1. ~~EasyOPD 的 SFT 入口（`examples/sft/`）是否能直接对接 Lightning-OPD 论文 §3.2 推荐的 3000-step、lr=8e-5、packing-enabled 配方？~~ **已决议**：不直接对接通用 SFT 入口；把配方翻译进 `easyopd/config/lightning_opd/sft.yaml`，由 `examples/lightning_opd_trainer/tools/run_sft.sh` 调用 verl `fsdp_sft_trainer` 加载该 yaml，关键字段由 `test_sft_config.py` 守护。剩余子问题：LlamaFactory 与 verl SFT 之间是否存在无法 1:1 翻译的字段（如 packing 的具体实现差异、deepspeed config）？这些差异在 Phase 4 实现 sft.yaml 时逐项 inline 注释解释。
2. ~~是否要在 `easyopd/methods/lightning_opd/` 暴露一个统一的 Python pipeline orchestrator（类似 `python -m easyopd.methods.lightning_opd.run_all`），还是只保留 shell 入口？~~ **已决议**：与其他方法（ROPD、SOD）保持一致，只保留 shell 入口。不新增 `__main__.py` 或 Python pipeline orchestrator；用户通过 `examples/lightning_opd_trainer/train_lightning_opd.sh` 和 `tools/*.sh` 启动，底层调用 `python3 -m verl.trainer.main_ppo` + Hydra override，与 ROPD/SOD 的启动方式完全一致。
3. ~~teacher logprob 预计算工具是否需要支持「分布式 prepare」（多机分片）？源仓库默认单机 8 GPU，第一阶段是否够用？~~ **已决议**：第一阶段默认单机 8 GPU，不做多机分片；且第一阶段只跑 CPU 测试（mock teacher URL），不实际调用 vLLM GPU 推理。多机分片作为后续优化项。
4. ~~30B-A3B 配置是否在第一阶段交付 dry-run 命令，还是延后到 Phase 5？~~ **已决议**：30B-A3B 不是必须交付项。Lightning-OPD 方法包应做到模型无关——换任意模型都能跑，不限于 Qwen3 系列。第一阶段以 4B/8B 为参考配置验证流程正确性，不单独为 30B-A3B 做适配。
5. ~~是否在 EasyOPD 根 README 中加入 Lightning-OPD 入口表项，还是只在 `examples/lightning_opd_trainer/README.md` 与 `docs/algo/lightning_opd.md` 中说明？~~ **已决议**：不在根 README 中加入；仅在 `examples/lightning_opd_trainer/README.md` 和 `docs/algo/lightning_opd.md` 中说明。
6. parquet schema 中 `teacher_log_probs` 列名是否与 EasyOPD 现有数据 schema 已存在的列冲突？**待确认**：实现阶段做全仓扫描后再定。
7. 是否需要为 Lightning-OPD 准备一份 EasyOPD 自带的 toy parquet（10 条样本 + 假 teacher logprob），用于让用户 dry-run 训练命令时实际跑通 1 step？**待确认**：实现阶段再决定。

## 完成定义

本次集成完成时，应满足：

- Lightning-OPD 核心代码位于 `easyopd/methods/lightning_opd/`。
- 用户入口位于 `examples/lightning_opd_trainer/`（含 `train_lightning_opd.sh` 与 `tools/`）。
- 配置位于 `easyopd/config/lightning_opd/`（至少含 `base.yaml`、`training.yaml`、`data_prep.yaml`、`sft.yaml`）。
- 默认 advantage estimator `on_policy_distillation` 通过方法包 `register()` 注册到 verl。
- 不引入 `slime/` 或 `slime_plugins/` 任何源代码。
- verl 侧改动总行数 ≤ 20，全部用 `# [EasyOPD:lightning_opd]` marker 包裹。
- 不修改 `pyproject.toml` / `requirements*.txt` / `setup.py` 的主线依赖。
- 测试位于 `tests/easyopd/lightning_opd/`，CPU 单测全部通过。
- `LIGHTNING_OPD_DRYRUN=true bash examples/lightning_opd_trainer/train_lightning_opd.sh` 退出 0 并打印完整命令预览。
- `LIGHTNING_OPD_DRYRUN=true bash examples/lightning_opd_trainer/tools/prepare_data.sh` 退出 0 并打印 Phase 1 + Phase 2 命令预览。
- `docs/algo/lightning_opd.md` 中文文档覆盖方法定义、teacher consistency 约束、数据流水线表、配置与环境变量、与论文结果的差异说明。
- `examples/lightning_opd_trainer/README.md` 覆盖完整 7 步 pipeline（含 SFT 步指向 `examples/sft/`）。
- 目标侧不存在 `slime\.`、`is_offline_opd`、`is_lightning_opd`、`SLIME_*` 命名（test_no_legacy_names.py 守护）。
- 目标侧不迁移源仓库的 `assets/`、`data/`、`checkpoints/`、`run_docker.sh`、`train.py`、`configs/lightning_opd/*.py`、`configs/opd/*.py`、`configs/models/*.sh`、`configs/sft/*.yaml`、`slime/`、`slime_plugins/`。
- 第一阶段不要求端到端 GPU 训练通过；GPU 验收在 Phase 5 独立 spec 中跟进。
