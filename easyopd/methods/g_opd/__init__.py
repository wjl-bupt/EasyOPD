"""
G-OPD: Generalized On-Policy Distillation with Reward Extrapolation.

This method implements the G-OPD framework which introduces a reward scaling factor
and a flexible reference model to generalize standard OPD. Building on G-OPD,
ExOPD (On-Policy Distillation with Reward Extrapolation) outperforms standard OPD
in both same-size and strong-to-weak distillation settings.

Paper: https://arxiv.org/abs/2602.12125

Integration mode: Mode A (lightweight verl modification)
    - Core algorithm in easyopd/methods/g_opd/core.py
    - Reference input utilities in easyopd/methods/g_opd/ref_input_utils.py
    - Config fields added to verl/workers/config/actor.py (PolicyLossConfig)
    - Advantage computation added to verl/workers/actor/dp_actor.py
    - Ref model input preparation added to verl/trainer/ppo/ray_trainer.py

Modified verl files:
    - verl/workers/config/actor.py: Added G-OPD fields to PolicyLossConfig
    - verl/workers/actor/dp_actor.py: Added G-OPD advantage computation branch
    - verl/trainer/ppo/ray_trainer.py: Added context distillation + base model log prob computation
    - verl/trainer/config/algorithm.py: Added G-OPD algorithm config fields
"""

from easyopd.methods.g_opd.core import (
    compute_g_opd_advantages,
    compute_multi_teacher_advantages,
    compute_standard_opd_advantages,
)
from easyopd.methods.g_opd.ref_input_utils import (
    prepare_ref_model_inputs,
    prepare_critique_distillation_inputs,
    prepare_ref_model_inputs_based_on_correct_solution,
)
from easyopd.registry import register_method

__all__ = [
    "compute_g_opd_advantages",
    "compute_multi_teacher_advantages",
    "compute_standard_opd_advantages",
    "prepare_ref_model_inputs",
    "prepare_critique_distillation_inputs",
    "prepare_ref_model_inputs_based_on_correct_solution",
    "GOPDMethod",
]


@register_method("g_opd")
class GOPDMethod:
    """G-OPD: Generalized On-Policy Distillation with Reward Extrapolation.

    Metadata class for the EasyOPD registry.
    """

    # Method metadata
    name = "g_opd"
    description = "G-OPD: Generalized On-Policy Distillation with Reward Extrapolation"
    paper_url = "https://arxiv.org/abs/2602.12125"

    # Files modified in verl
    verl_modified_files = [
        "verl/workers/config/actor.py",          # Added G-OPD fields to PolicyLossConfig
        "verl/workers/actor/dp_actor.py",        # Added G-OPD advantage computation
        "verl/trainer/ppo/ray_trainer.py",       # Added context distillation + base model log probs
        "verl/trainer/config/algorithm.py",      # Added G-OPD algorithm config fields
    ]
