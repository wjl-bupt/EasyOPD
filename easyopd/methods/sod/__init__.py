"""
SOD: Step-wise On-policy Distillation for Small Language Model Agents.

This method implements step-wise adaptive re-weighting of OPD signals
to prevent cascade failures in tool-integrated reasoning (TIR) scenarios.

Paper: https://arxiv.org/abs/2605.07725

Integration mode: Mode A (lightweight verl modification)
    - Core algorithm in easyopd/methods/sod/core.py
    - Config fields added to verl/trainer/config/algorithm.py
    - If-branch added to verl/trainer/ppo/ray_trainer.py

Modified verl files:
    - verl/trainer/config/algorithm.py: Added TokenKLRegConfig dataclass
    - verl/trainer/ppo/ray_trainer.py: Added SOD if-branch after advantage computation
"""

from easyopd.methods.sod.core import (
    apply_stepwise_opd,
    compute_stepwise_opd_weights,
)
from easyopd.registry import register_method

__all__ = [
    "compute_stepwise_opd_weights",
    "apply_stepwise_opd",
    "SODMethod",
]


@register_method("sod")
class SODMethod:
    """SOD: Step-wise On-policy Distillation.

    Metadata class for the EasyOPD registry.
    """

    # Method metadata
    name = "sod"
    description = "SOD: Step-wise On-policy Distillation for Small LM Agents"
    paper_url = "https://arxiv.org/abs/2605.07725"

    # Files modified in verl
    verl_modified_files = [
        "verl/trainer/config/algorithm.py",      # Added TokenKLRegConfig
        "verl/trainer/ppo/ray_trainer.py",        # Added SOD if-branch
    ]
