# Listwise-KL On-Policy Distillation

> 改进对象:**标准 OPD**(`opd/scripts/baselines/opd.sh`,逐 token reverse-KL 的
> policy-gradient 蒸馏)。**不涉及 relay handoff**;只借用本仓库的 teacher-loop /
> distillation 基础设施。本文件是方法定义与落地设计说明,不含代码。

## 1. 一句话概括

在 student 采样的 **K 条轨迹**上,让 **teacher 与 student 各自把这 K 条的序列似然
softmax 归一化成一个组内 categorical 分布**,再最小化 teacher 分布到 student 分布的
KL。直觉:把 teacher 对这 K 条候选的**相对偏好排序**蒸馏给 student。

- 纯 **on-policy**:候选全部来自当前 student,不做 off-policy replay。
- **不带** importance correction:目标/学生分布都用**原始序列 logprob**,不除以行为
  策略 $b$。这与"$q_T\propto t/b$"的版本是**不同**的方法(区别见 §5、§9)。

## 2. 记号

固定 prompt $x$,student rollout 采 $K$ 条完整轨迹 $\tau_1,\dots,\tau_K\sim\pi_\theta(\cdot\mid x)$(即 `rollout.n=K`)。对每条轨迹定义序列级 logprob(仅在 response token 上求和):

| 符号 | 含义 | 来源 |
| --- | --- | --- |
| $s_i^{S}=\sum_t\log\pi_\theta(a_t\mid y_{<t})$ | 训练态 student,**可导** | actor 前向 `model_output["log_probs"]` 求和 |
| $s_i^{b}=\sum_t\log\pi_{b}(a_t\mid y_{<t})$ | rollout 快照 $b$,**stop-grad** | `data["old_log_probs"]` 求和 |
| $s_i^{T}=\sum_t\log\pi_{T}(a_t\mid y_{<t})$ | teacher 强制解码打分 | teacher-loop 的 `teacher_token_logprobs` 在 response mask 上求和 |
| $R_i$ | 可选任务 reward | 现有 math reward |
| $\lvert\tau_i\rvert$ | response 长度 | response mask 计数 |

纯 on-policy 下,更新起点 $\pi_\theta=\pi_b$,故 $s_i^{S}=s_i^{b}$;组内漂移交给 PG loss 的 PPO ratio 处理(见 §4)。

## 3. 目标函数

**打分**(可选长度归一化 $\hat s_i=s_i/\lvert\tau_i\rvert$,温度 $\beta$):

$$
u_i=\beta\,\hat s_i^{\,T}+\eta\,R_i\quad(\text{target}),\qquad
v_i=\beta\,\hat s_i^{\,S}\quad(\text{student}).
$$

**组内**(同一 prompt 的 K 条)softmax 归一化成分布:

$$
q_T(i)=\frac{e^{u_i}}{\sum_{j=1}^{K}e^{u_j}},\qquad
q_S(i)=\frac{e^{v_i}}{\sum_{j=1}^{K}e^{v_j}}.
$$

**每组 loss**:

$$
\boxed{\;\mathcal L=D_{\mathrm{KL}}\!\big(q_T\,\|\,q_S\big)
=\sum_{i=1}^{K}q_T(i)\log\frac{q_T(i)}{q_S(i)}\;}
$$

- $\eta=0$:纯 teacher 排序匹配(**默认**,对应"改进纯 OPD")。
- $\eta>0$:reward-aware —— 把任务 reward 加进 teacher 目标分数后再归一化。
- $\beta$、长度归一化、$\eta$ 均为开关(见 §7)。

## 4. 梯度 → 逐序列 policy gradient(精确)

对 $\mathcal L$ 求梯度($q_S$ 是 $\theta$ 的 softmax,归一化项梯度恰好给出 $q_S(i)$ 系数,并用 $\sum_i q_T(i)=1$):

$$
\nabla_\theta\mathcal L
=\sum_{i=1}^{K}\big[q_S(i)-q_T(i)\big]\,\nabla_\theta\log \pi_\theta(\tau_i).
$$

因此下降方向就是一个**逐序列 REINFORCE**,advantage 为

$$
\boxed{\;A_i=q_T(i)-q_S(i)\;}
$$

把 $A_i$ **广播到该轨迹的 response token**,喂给标准 PG loss(最大化 $\sum_i A_i\log\pi_\theta(\tau_i)$)即复现 $-\nabla_\theta\mathcal L$。

要点:

- **baseline 是 $q_S(i)$**(student 自己的 softmax 偏好),**不是** $1/K$。$1/K$ 只是"带 $/b$ 版本"在 on-policy 下的特例;本方法不带 $/b$,故 $q_S$ 一般不均匀。
- $\sum_i q_T(i)=\sum_i q_S(i)=1\Rightarrow\sum_i A_i=0$,advantage 天然 mean-zero(无需再减均值)。
- 直觉:teacher 比 student 当前更偏好的轨迹($q_T>q_S$)→ $A_i>0$ → 抬高其概率;反之压低。
- **on-policy 实现**:起点 $\pi_\theta=\pi_b$,可直接用 `old_log_probs` 预计算 $q_S(i)=\mathrm{softmax}(\beta\hat s^{\,b})$(与 $q_T$ 同为 batch 级、按 uid 分组的 detached 标量)。一次更新内的策略漂移由 PG loss 的 PPO ratio $\pi_\theta/\pi_b$ 隐式修正 —— 与 GRPO 完全同款近似(仅在每次更新起点精确)。

## 5. 与 GRPO / 标准 OPD 的关系

| | 分组 | 每序列 advantage | 粒度 |
| --- | --- | --- | --- |
| GRPO | uid | 归一化 task reward $(R_i-\bar R)/\sigma$ | 序列(广播到 token) |
| **本方法** | uid | $q_T(i)-q_S(i)$(teacher 与 student 的组内 softmax 偏好之差) | 序列(广播到 token) |
| 标准 OPD | — | 逐 token 负 reverse-KL | token |

**结论:结构与 GRPO 同构**,只是把"组内归一化 reward"换成"组内 teacher-vs-student softmax 偏好差"。这让它能直接挂进 verl 的 advantage-estimator 抽象。

## 6. verl 落地映射

1. **Rollout**:`rollout.n=K`(现 `opd.sh` 为 1),纯 student vLLM rollout,**不启用 relay patch**。
2. **Teacher**:`distillation.enabled=True` 仅用于启动 teacher-loop 拿 teacher logprob。
3. **新增 advantage estimator**(`register_adv_est`,如 `listwise_kl`):
   - 输入:按 `uid` 分组的 $s_i^{T}$、$s_i^{b}$、可选 $R_i$、$\lvert\tau_i\rvert$。
   - 组内计算 $q_T,q_S$,输出 $A_i=q_T(i)-q_S(i)$,广播到 response token。
   - 复用现有 `compute_grpo_outcome_advantage` 的 uid 分组模式(`index=non_tensor_batch["uid"]`)。
4. **Loss**:复用现有 `use_policy_gradient` 的 PG 路径,不改 loss 内部。
5. **数据接线**:需把 teacher 序列 logprob(`teacher_token_logprobs` 在 response mask 上求和)与 `old_log_probs` 的序列和放到 batch,供 estimator 使用。

> 改动集中在"**新增一个 advantage estimator + 一个基于 `opd.sh` 的新脚本**",不重写 loss、不动 relay。

## 7. 配置开关

| 开关 | 含义 | 默认 |
| --- | --- | --- |
| `K` (`rollout.n`) | 每 prompt 候选数 | 8(待定) |
| `beta` | softmax 温度(作用于序列 logprob) | 1.0 |
| `length_norm` | 序列 logprob 是否按 token 数平均 | 开(缓解长短失衡)|
| `eta` | reward 权重($\eta=0$=纯 teacher) | 0 |
| `kl_direction` | $D_{\mathrm{KL}}(q_T\|q_S)$ / 反向 | `qT_to_qS` |
| `std_norm` | advantage 是否再除组内 std | 关(Dr.GRPO 式)|

## 8. 性质与局限(诚实声明)

- **仅在 student 支持集上重排**:候选全部来自 student、无 $/b$ 修正 ⇒ **不会**出现 out-of-support 的 importance 爆炸,但也**无法注入 student 采不到的 teacher 优质模式**(coverage 上限)。这正是"student rollout 落在 teacher 支持集外"讨论里的**方向 B** 未被解决的部分。
- **收敛语义**:组内 $q_S\to q_T$,即 student 在**自身样本**上的相对似然逼近 teacher 的相对似然 ⇒ 属于**分布/排序一致性保证**,**不是**任务 reward 的逐步单调改进保证(此前的反例逻辑仍然适用)。
- **teacher 信号强度**:若 K 条的 teacher logprob 相近($q_T$ 近均匀),advantage 退化为 $-(q_S-\tfrac1K)$,只起到"抑制 student 过度自信"的作用。$\beta$ 与长度归一化直接影响信号强弱,需实验标定。

## 9. 与"$q_T\propto t/b$"版本的区别(备忘)

早期讨论过带 proposal correction 的版本 $q_T\propto t/b,\;q_S\propto p_\theta/b$,其 on-policy advantage 化简为 $A_i=q_T(i)-\tfrac1K$。**当前方法不采用它**:本方法目标/学生分布都用**原始序列 logprob** 的 softmax,baseline 因此是 $q_S(i)$ 而非 $1/K$。两者目标不同——带 $/b$ 逼近的是全局 $p_\theta=t$(需 coverage),不带 $/b$ 逼近的是**组内相对排序一致**。

## 10. 待定 / 下一步

工程细节见 §11。开工前必须先落地的一件事:**确认在线 teacher 序列 logprob 能在
`compute_advantage` 之前 materialize 到 batch**(§11.2),这是整个方案能否走
advantage-estimator 路线的前提。其余待标定项:$\beta$ 与长度归一化的组合、`K` 默认值。

---

## 11. 工程实现(verl 具体改动)

### 11.1 为什么必须在 batch 层预计算 advantage

$q_T,q_S$ 是**组内(同一 prompt 的 K 条)** softmax,需要同组 K 条的序列 logprob 同时在场。
而 actor 更新按 microbatch/`dynamic_bsz` 切分、且跨 DP rank,**同一 prompt 的 K 条不保证在同一
microbatch**。因此组内 softmax **不能**在 per-microbatch 的 loss 里算。

解决:利用 §4 的化简,把 $A_i=q_T(i)-q_S(i)$ 作为**每序列标量**在 **batch 层**(此处
全 K 条与 `uid` 都在)一次性算好、detach、广播到 response token,再走**现成的逐 token PG**。
纯 on-policy 下 $q_S$ 用 `old_log_probs` 计算(起点 $\pi_\theta=\pi_b$ 精确;组内漂移由 PG 的
PPO ratio 修正)。这样**完全绕开跨-microbatch 耦合**。

### 11.2 数据流:三个序列级量必须在 advantage 之前就位

`compute_advantage`(`verl/trainer/ppo/ray_trainer.py:187`)在 actor 更新前运行。它需要:

| 量 | 来源 | 状态 |
| --- | --- | --- |
| `index` = `uid` | `data.non_tensor_batch["uid"]`(GRPO 已用,ray_trainer.py:243/257) | ✅ 现成 |
| $s_i^{b}$ = old seq-logprob | `data.batch["old_log_probs"]` 在 mask 上求和(`_compute_old_log_prob`, ray_trainer.py:1467) | ✅ 现成 |
| $s_i^{T}$ = teacher seq-logprob | teacher-loop 逐 token teacher logprob 在 mask 上求和 | ⚠️ **需确认时机** |
| $R_i$(仅 $\eta>0$) | 现有 reward,`token_level_rewards.sum(-1)` | ✅ 现成 |

**关键确认项**:标准 OPD 里 teacher logprob 目前是喂给 distillation loss(在 actor 内、更新时)。
本方案要求它在 `compute_advantage` **之前**出现在 `data.batch[<teacher_key>]`。两条路:
- **(优先)** 若在线 teacher-logprob 已作为 batch 级步骤(类似 `_compute_old_log_prob` /
  `_compute_ref_log_prob`, ray_trainer.py:1443/1467)先行算好 → 直接读键(疑似
  `teacher_log_probs`,`compute_distillation_loss_reverse_kl_estimator`, losses.py:456 在用)。
- **(否则)** 新增一个 batch 级 `_compute_teacher_log_prob(batch)` 步骤,复用
  `AsyncTeacherLLMServerManager.compute_teacher_logprobs_single`(teacher_manager.py:102),把
  逐 token teacher logprob 存进 `batch.batch["teacher_log_probs"]`,置于 `compute_advantage` 前。

### 11.3 新增 advantage estimator

`verl/trainer/ppo/core_algos.py`:
1. `AdvantageEstimator` 枚举加 `LISTWISE_KL = "listwise_kl"`。
2. 仿 `compute_grpo_outcome_advantage`(core_algos.py:267)注册:

```python
@register_adv_est(AdvantageEstimator.LISTWISE_KL)
def compute_listwise_kl_advantage(
    token_level_rewards, response_mask, index,
    teacher_log_probs, old_log_probs, config=None,
):
    # 1) 序列级 logprob(仅 response)
    #    s_T = (teacher_log_probs * response_mask).sum(-1)
    #    s_b = (old_log_probs     * response_mask).sum(-1)
    # 2) 可选长度归一化:  / response_mask.sum(-1)
    # 3) 打分:  u = beta*s_T_hat + eta*token_level_rewards.sum(-1)
    #           v = beta*s_b_hat
    # 4) 按 index(uid)分组,组内 softmax(log-sum-exp 稳定)得 q_T, q_S
    # 5) A_i = q_T(i) - q_S(i)              # sum_i A_i = 0,天然 mean-zero
    # 6) 广播:  advantages = (A_i)[:, None] * response_mask
    # 返回 (advantages, advantages)  # returns 走 PG 时不用
```

`config`(`AlgoConfig`)承载 `beta / eta / length_norm / kl_direction / std_norm`。

`compute_advantage`(ray_trainer.py:187,`adv_kwargs` 构建处)加分支:当
`adv_estimator == "listwise_kl"` 时注入
`adv_kwargs["teacher_log_probs"] = data.batch["teacher_log_probs"]`(以及已有的
`old_log_probs`)。

### 11.4 loss 侧:复用现成 PG,不写新 loss

拿到 batch 级 `advantages` 后,actor 更新走**标准 policy-gradient**(与 GRPO 同路),
不需要 distillation loss 的梯度。`distillation.enabled=True` 仅用于**驱动 teacher-loop
产出 teacher logprob**。需要一个开关让"teacher logprob 只喂 advantage、不进 distillation
loss 的反传",避免和 §11.2 的 teacher 通道重复计算 / 双重反传。

> 备选(diff 更贴近 `opd.sh` 现状):把 §11.3 的每序列 `A_i` 预计算结果写进
> `data`,再加一个 thin `loss_mode=listwise_kl`,在 `distillation_loss()` 的
> `use_policy_gradient` 分支(losses.py:301)里读它当 advantage。工作量与 estimator 路线
> 相当,但复用了 `opd.sh` 已走通的 distillation-loss PG 管线。二选一在实现时定。

### 11.5 训练脚本

新增 `opd/scripts/listwise_kl/train.sh`,基于 `opd/scripts/baselines/opd.sh` 改:
- `actor_rollout_ref.rollout.n=${K}`(默认 8;`opd.sh` 为 1)——**这是产生候选组的关键**。
- `algorithm.adv_estimator=listwise_kl`。
- 新增 `+algorithm.listwise_kl.{beta,eta,length_norm,kl_direction,std_norm}` 覆盖(见 §7)。
- 保留 `distillation.enabled=True` + teacher 配置(拿 teacher logprob);关掉 k1 distillation-loss 反传(§11.4 开关)。
- 其余(lr 1e-6 常数、`ppo_epochs=1`、batch=128、reward manager)沿用 OPD 默认。

### 11.6 实现顺序(checklist)

1. **[阻塞前置]** 确认/实现:teacher 序列 logprob 在 `compute_advantage` 前进 `data.batch`(§11.2)。
2. `compute_listwise_kl_advantage` estimator + 枚举 + `AlgoConfig` 字段(§11.3)。
3. `compute_advantage` 注入 teacher/old logprob 的分支(§11.3)。
4. §11.4 开关:teacher logprob 走 advantage、不进 distillation loss 反传。
5. `train.sh`(§11.5)。
6. CPU 单测(仿 `tests/opd/`):给定小 batch 的 teacher/old seq-logprob + uid,校验
   $A_i=q_T-q_S$、$\sum_i A_i=0$、$\eta/\beta/$长度归一化开关行为。
7. 2-GPU(1 actor + 1 teacher)冒烟:小 `K`、短 response,确认 loss/adv 数值健康、能训。

### 11.7 数值与正确性注意

- 组内 softmax 用 **log-sum-exp**(序列 logprob 数量级大)。
- 序列 logprob 随长度线性增长:不做长度归一化时 $\beta$ 需相应调小,否则 softmax 饱和到单条。
- teacher 与 student 的 response mask / token 对齐必须一致(同一份采样轨迹),否则 $s^T,s^b,s^S$ 不可比。
- 单条候选组(某 prompt 只剩 1 条有效)退化:$q_T=q_S=1$、$A=0$,与 GRPO 单样本组处理一致(core_algos.py:315)。

---

## 12. 实际实现(as-built)与 §11 设计的偏差

§11 假定可以改 `verl/`。本次实现的约束是**只能改 `easyopd/methods/opld/`**,
因此走 estimator 路线(§11.3)但用**运行时 patch** 代替源码修改。`verl/` 零改动。

### 12.1 文件

| 文件 | 职责 |
| --- | --- |
| `core.py` | 纯张量数学:`compute_listwise_advantage`、`read_listwise_config`,不 import verl,可 CPU 单测(对应 §11.6 第 6 项) |
| `advantage_estimator.py` | `@register_adv_est("listwise")` + `compute_advantage` 包装器 |
| `__init__.py` | `@register_method("listwise")`(别名 `opld`)+ `register()` |
| `hooks.py` | 空 —— 见 §11.1,故意不提供 LossHook |

### 12.2 与 §11.3 的三处偏差

1. **不改 `AdvantageEstimator` 枚举。** `register_adv_est` 接受裸字符串,
   `compute_advantage` 的 generic 分支用 `get_adv_estimator_fn(adv_estimator)`
   查表,枚举成员非必需(`lightning_opd` 已证明)。注册名为 `listwise`
   (非 `listwise_kl`)。
2. **不改 `compute_advantage` 源码注入 kwargs。** 该函数的 generic 分支只传
   `token_level_rewards / response_mask / config / index`,`teacher_log_probs`
   仅对硬编码名 `on_policy_distillation`(lightning_opd 占用)转发。改为在
   `register()` 里包装 `ray_trainer.compute_advantage`,命中 `listwise` 时自行
   完成序列级归约并调用 estimator,其余名字原样透传。
3. **配置不放 `AlgoConfig.listwise_kl`。** `AlgoConfig` 是 frozen dataclass,
   未声明的字段会被 `BaseConfig.get` 的 `except AttributeError` 静默吞掉,
   yaml 配置**永远不生效**。改用已声明的 `AlgoConfig.easyopd` 字段,路径为
   `algorithm.easyopd.listwise`。未知 key 会**报错**而非忽略。

### 12.3 §11.2 阻塞前置项的现状

§10/§11.6 把"teacher 序列 logprob 能否在 `compute_advantage` 前 materialize"
列为开工阻塞项。本仓库的实测结论:

| 键 | 注入位置 | 相对 `compute_advantage`(ray_trainer.py:2484) |
| --- | --- | --- |
| `teacher_log_probs`(lightning_opd 离线 parquet) | ray_trainer.py:2428 | **之前** ✅ 可用 |
| `opsa_teacher_log_probs`(在线 frozen-ref forward) | ray_trainer.py:2564 | **之后** ⚠️ 取不到 |

estimator 按上表顺序查找。**离线路线开箱可用;在线路线需要把 teacher forward
上提到 `adv` 块之前**,那是 `verl/` 改动,超出本次范围。缺 teacher logprob 时
抛 `OPLDMissingTeacherLogprobs` 并说明原因,不静默退化。

### 12.4 §11.4 的"双重反传"开关

不需要。本实现不注册任何 distillation loss,teacher logprob 只经 advantage
进入训练,梯度全部由现成 PG 路径产生,不存在重复计算或双重反传。

### 12.5 §7 反向 KL 开关的修正

`kl_direction: qS_to_qT` 实现为真正的反向 KL 梯度:

$$A_i=-q_S(i)\,\big(w_i-D_{\mathrm{KL}}(q_S\|q_T)\big),\qquad w_i=\log q_S(i)-\log q_T(i).$$

参考实现里该分支写的是 `q_S - q_T`,即前向分支的**相反数**(等于对前向 KL 做梯度
*上升*),并非反向 KL。此处未沿用。

### 12.6 已知近似

`length_norm` 只归一化 softmax 的 logits。下游 PG 求导的是 $\log\pi$ 而非
$\log\pi/L$,严格推导下 $A_i$ 还应带 $1/L_i$ 因子。沿用"组内长度可比"的常见近似
(与参考实现一致)。

### 12.7 配置示例

```yaml
easyopd:
  method:
    name: listwise      # 必填!见下方说明

algorithm:
  adv_estimator: listwise
  easyopd:
    listwise:
      beta: 1.0
      eta: 0.0
      length_norm: true
      kl_direction: qT_to_qS
      std_norm: false

actor_rollout_ref:
  rollout:
    n: 8          # 必须 > 1,否则每组单样本、A ≡ 0
```

`rollout.n=1` 时超过 50% 的组退化会打 warning(§11.7 最后一条)。

> **`easyopd.method.name: listwise` 不可省略。**
> 注册链路是:`RayPPOTrainer.__init__`(ray_trainer.py:480)→
> `HookDispatcher.from_config` → 仅当能从 config 解析出 method name 时才调
> `ensure_discovered()`(hook_dispatch.py:128)→ import 本包 → 注册 estimator +
> 装 patch。若只设 `algorithm.adv_estimator=listwise` 而不设 method name,
> discovery 不触发,`get_adv_estimator_fn("listwise")` 会抛
> `Unknown advantage estimator simply: listwise`。

### 12.8 单测

```bash
pytest easyopd/methods/opld/test_core.py
```

纯 CPU,只依赖 torch + numpy,不需要 verl / ray / GPU。覆盖 §11.6 第 6 项:
组内 mean-zero、$A_i=q_T-q_S$ 闭式、mask 正确性、单样本组退化、$\beta$ /
`length_norm` / `eta` / `std_norm` 开关、反向 KL 不等于前向取负、配置错拼报错。


