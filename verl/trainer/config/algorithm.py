# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "TokenKLRegConfig", "RolloutCorrectionConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", "reward_std", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
        reward_std_threshold (float): Minimum standard deviation required when metric is "reward_std".
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0
    reward_std_threshold: float = 0.0


@dataclass
class TokenKLRegConfig(BaseConfig):
    """Configuration for token-level KL regularizer."""

    enable: bool = False
    coef: float = 0.0
    gamma: float = 1.0
    beta_min: float = 0.0
    beta_max: Optional[float] = None
    # [EasyOPD] Step-wise OPD parameters (consumed by SOD method via hook dispatch)
    stepwise_enable: bool = False
    stepwise_epsilon: float = 1e-6
    stepwise_delta: float = 0.5
    stepwise_opd_coef: float = 1.0


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: FilterGroupsConfig = field(default_factory=FilterGroupsConfig)
    token_kl_reg: TokenKLRegConfig = field(default_factory=TokenKLRegConfig)
    # ============ [EasyOPD] Context distillation and ref solution distillation ============
    critique_vllm_url: Optional[str] = None
    critique_model: Optional[str] = None
    max_critique_tokens: int = 2048
    critique_temperature: float = 0.0
    critique_top_p: float = 1.0
    use_ref_solution_distillation: bool = False
    # ============ [EasyOPD] End ============
    # ============ [EasyOPD] Rollout correction configuration ============
    rollout_correction: Optional["RolloutCorrectionConfig"] = None
    # ============ [EasyOPD] End ============
    # ============ [EasyOPD] Generic method config (consumed by HookDispatcher) ============
    easyopd: Optional[dict[str, Any]] = None
    # ============ [EasyOPD] End ============


# ============ [EasyOPD] Rollout correction config ============
@dataclass
class RolloutCorrectionConfig(BaseConfig):
    """Configuration for Rollout Correction (addresses off-policy issues in RL training).

    Rollout Correction handles off-policiness from multiple sources:
    1. Policy mismatch: Rollout policy (e.g., vLLM BF16) vs Training policy (e.g., FSDP FP32)
    2. Model update staleness: Rollout data collected from older policy checkpoints

    Args:
        rollout_is (Optional[str]): IS weight aggregation level.
            - None: No IS weights
            - "token": Per-token IS weights (low variance, biased)
            - "sequence": Per-sequence IS weights (unbiased, high variance)
        rollout_is_threshold (float): Upper threshold for IS weight truncation.
        rollout_rs (Optional[str]): Rejection sampling mode.
        rollout_rs_threshold (Optional[str | float]): Threshold for rejection sampling.
    """

    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: float = 2.0
    rollout_is_batch_normalize: bool = False
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[Any] = None
# ============ [EasyOPD] End ============
