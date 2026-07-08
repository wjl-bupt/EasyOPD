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

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = ["SelfDistillationConfig", "PolicyLossConfig", "ActorConfig", "FSDPActorConfig", "McoreActorConfig"]


# ============ [EasyOPD] Self-distillation configuration ============
@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss (Vision-OPD).

    Distillation is enabled when policy_loss.loss_mode is "vopd".

    Args:
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        gamma (float): Weight on the self-distillation loss for Vision-OPD
            (``policy_loss = vopd_loss * gamma + grpo_loss``). NOT used by SDPO
            (``loss_mode='sdpo'``), whose loss has no GRPO-fallback term.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region", "progressive".
        teacher_update_rate (float): EMA update rate for teacher weights.
        teacher_update_interval (Optional[int]): Hard-sync teacher every N steps (progressive mode).
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        teacher_always_on (bool): Whether to distill every sample directly from teacher.
        teacher_model_source (str): Teacher source. Options: "legacy", "current" or "fixed".
        teacher_model_path (Optional[str]): Fixed teacher model path when teacher_model_source="fixed".
        teacher_image_key (Optional[str]): Dataset column holding teacher-side images.
        teacher_prompt_mode (Optional[str]): Teacher prompt mode. Options: None, "answer_hint".
        fallback_to_policy_loss_on_missing_teacher (bool): Fall back to vanilla policy loss when teacher unavailable.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0
    gamma: float = 1.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    teacher_update_interval: Optional[int] = None
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    # Templates mirror lasgroup/SDPO actor.yaml exactly (YAML `|-` strips the
    # trailing newline), so the self-teacher reprompt text is byte-faithful.
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question."
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    teacher_always_on: bool = False
    teacher_model_source: str = "legacy"
    teacher_model_path: Optional[str] = None
    teacher_image_key: Optional[str] = None
    teacher_prompt_mode: Optional[str] = None
    answer_hint_template: str = (
        "\n\nHere is a reference solution to this problem:\n"
        "{answer}\n\n"
        "After understanding the reference solution, please try to solve this problem using your own approach below:\n"
    )
    fallback_to_policy_loss_on_missing_teacher: bool = False
    log_prob_dump_dir: Optional[str] = None
# ============ [EasyOPD] End ============


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg', 'vopd'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
        only_reverse_kl_advantages (bool): [EasyOPD] Use reverse KL as advantages for OPD.
        lambda_vals (float): [EasyOPD] Reward scaling factor (1.0=OPD, >1.0=ExOPD).
        multi_teacher_distill (bool): [EasyOPD] Enable multi-teacher distillation.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    # ============ [EasyOPD] On-policy distillation parameters ============
    only_reverse_kl_advantages: bool = False
    lambda_vals: float = 1.0
    multi_teacher_distill: bool = False
    # ============ [EasyOPD] End ============
    # ============ [EasyOPD:Simple] Cross-tokenizer KD parameters ============
    simple_kl_direction: str = "reverse"  # 'forward' or 'reverse'
    simple_loss_clamp: float = 10.0  # Max clamp value for per-token KL
    # ============ [EasyOPD:Simple] End ============


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        entropy_coeff (float): Entropy coefficient for regularization.
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0
    tis_imp_ratio_cap: float = -1
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    # ============ [EasyOPD] KL distillation config (used by OPCD method) ============
    kl_topk: int = 0  # If > 0, use top-k logits for full KL computation (memory efficient)
    kl_renorm_topk: bool = False  # Whether to renormalize top-k log-probs before KL
    profile_kl: bool = False  # Whether to profile KL computation
    # ============ [EasyOPD] End ============
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    data_loader_seed = 1
    n: int = 1  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    # ============ [EasyOPD] Self-distillation config ============
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)
    # ============ [EasyOPD] End ============

    # ============ [EasyOPD:OPSA] On-Policy Self-Distillation for Safety Alignment ============
    opsa_enable: bool = False
    opsa_temperature: float = 1.0
    opsa_window_size: int = 32
    opsa_decay_type: str = "linear"  # Options: linear, step, exponential
    opsa_min_weight: float = 0.1
    opsa_use_window_weighting: bool = True
    opsa_distillation_loss_coef: float = 1.0
    opsa_loss_agg_mode: str = "token-mean"
    opsa_harmful_context: str = (
        "You are a helpful and safe AI assistant. When you encounter a harmful, "
        "dangerous, or unethical request, you MUST refuse clearly and explain why. "
        "Safety is your highest priority."
    )
    opsa_benign_context: str = (
        "You are a helpful AI assistant. Respond to the user's request directly "
        "and helpfully. Do not over-refuse safe and reasonable requests."
    )
    opsa_kl_type: str = "mixed"  # Options: forward, reverse, mixed
    opsa_mixed_kl_weight: float = 0.5  # Weight for forward KL in mixed mode
    opsa_topk_logits_k: int = 512  # Top-K logits for KL computation (0 = per-token only)
    opsa_tfr_eval_frequency: int = 10
    opsa_tfr_threshold: float = 0.8
    # ============ [EasyOPD:OPSA] End ============

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        data_loader_seed (Optional[int]): Seed for data loader. If None, uses global seed.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
