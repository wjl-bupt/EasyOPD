"""
SDPO: Self-Distilled Policy Optimization (Reinforcement Learning via Self-Distillation)

This method implements the SDPO framework which augments on-policy optimization
with self-distillation from the model's own high-reward trajectories.

Paper: "Reinforcement Learning via Self-Distillation"
       Hübotter et al., 2026
       https://arxiv.org/abs/2601.20802

Integration mode: Mode A (lightweight verl modification)
    - Core algorithm in easyopd/methods/sdpo/core.py
    - SDPO loss triggered by setting actor.policy_loss.loss_mode = "sdpo"
    - Self-distillation config in actor.self_distillation
    - Reprompting logic builds teacher prompts from successful demonstrations

Modified verl files:
    - verl/trainer/ppo/core_algos.py: Register compute_self_distillation_loss function
    - verl/workers/config/actor.py: Add SelfDistillationConfig dataclass
    - verl/workers/actor/dp_actor.py: Add self-distillation forward pass in update_policy
    - verl/trainer/ppo/ray_trainer.py: Add _maybe_build_self_distillation_batch method
"""

from easyopd.methods.sdpo.core import (
    compute_sdpo_self_distillation_loss,
    build_reprompt_text,
    select_demonstration,
    compute_ema_update,
)
from easyopd.registry import register_method

__all__ = [
    "compute_sdpo_self_distillation_loss",
    "build_reprompt_text",
    "select_demonstration",
    "compute_ema_update",
    "SDPOMethod",
]


@register_method("sdpo")
class SDPOMethod:
    """SDPO: Self-Distilled Policy Optimization.

    Metadata class for the EasyOPD registry.
    """

    # Method metadata
    name = "sdpo"
    description = (
        "SDPO: Self-Distilled Policy Optimization. "
        "Augments on-policy RL with self-distillation from the model's own "
        "high-reward trajectories, converting feedback into dense learning signals "
        "without any external teacher."
    )
    paper_url = "https://arxiv.org/abs/2601.20802"
    code_url = "https://github.com/lasgroup/SDPO"

    # Files modified in verl
    verl_modified_files = [
        "verl/trainer/ppo/core_algos.py",       # Add compute_self_distillation_loss
        "verl/workers/config/actor.py",          # Add SelfDistillationConfig, PolicyLossConfig.sdpo
        "verl/workers/actor/dp_actor.py",        # Add self-distillation forward in update_policy
        "verl/trainer/ppo/ray_trainer.py",       # Add _maybe_build_self_distillation_batch
    ]
