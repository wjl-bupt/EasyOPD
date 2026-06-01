"""
OPSA: Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation

Paper: https://arxiv.org/abs/2605.15239
Code: https://github.com/FYYFU/OPSA
Integration mode: Mode A (lightweight verl modification)
    - Core algorithm in easyopd/methods/opsa/core.py
    - Config fields added to verl/workers/config/actor.py
    - If-branch added to verl/trainer/ppo/ray_trainer.py

Modified verl files:
    - verl/workers/config/actor.py: Added OPSAConfig dataclass
    - verl/workers/actor/dp_actor.py: Added OPSA per-token KL loss with privileged contexts
    - verl/trainer/ppo/ray_trainer.py: Added privileged context injection logic
"""

from easyopd.methods.opsa.core import (
    compute_teacher_flip_rate,
    compute_opsa_kl_loss,
    compute_early_window_weights,
    opsa_loss,
)
from easyopd.registry import register_method

__all__ = [
    "compute_teacher_flip_rate",
    "compute_opsa_kl_loss",
    "compute_early_window_weights",
    "opsa_loss",
    "OPSAMethod",
]


@register_method("opsa")
class OPSAMethod:
    """OPSA: On-Policy Self-Distillation for Safety Alignment.

    Metadata class for the EasyOPD registry.
    """

    name = "opsa"
    description = (
        "OPSA: On-Policy Self-Distillation for Safety Alignment. "
        "Reduces the safety tax in LLM alignment by concentrating dense "
        "per-token KL supervision on safety-critical decision windows, "
        "using type-conditional privileged contexts and Teacher Flip Rate (TFR) "
        "for context selection."
    )
    paper_url = "https://arxiv.org/abs/2605.15239"
    code_url = "https://github.com/FYYFU/OPSA"

    verl_modified_files = [
        "verl/workers/config/actor.py",       # Added OPSAConfig dataclass
        "verl/workers/actor/dp_actor.py",     # Added OPSA per-token KL loss
        "verl/trainer/ppo/ray_trainer.py",    # Added privileged context injection
    ]
