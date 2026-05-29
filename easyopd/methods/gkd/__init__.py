"""
GKD: Generalized Knowledge Distillation (On-Policy Distillation of Language Models)

This method implements the GKD framework which combines on-policy sampling with
dense teacher feedback using a Generalized Jensen-Shannon Divergence (JSD).

Paper: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
       Agarwal et al., ICLR 2024
       https://arxiv.org/abs/2306.13649

Integration mode: Mode A (lightweight verl modification)
    - Core algorithm in easyopd/methods/gkd/core.py
    - GKD loss registered in verl/trainer/distillation/losses.py via register_distillation_loss
    - GKD config fields (beta) added to verl/workers/config/distillation.py
    - On-policy distillation is already supported by verl's existing infrastructure

Modified verl files:
    - verl/trainer/distillation/losses.py: Register GKD JSD loss mode
    - verl/workers/config/distillation.py: Add gkd_beta field to DistillationLossConfig
"""

from easyopd.methods.gkd.core import (
    generalized_jsd,
    generalized_jsd_from_estimator,
    gkd_loss,
    compute_on_policy_ratio,
)
from easyopd.registry import register_method

__all__ = [
    "generalized_jsd",
    "generalized_jsd_from_estimator",
    "gkd_loss",
    "compute_on_policy_ratio",
    "GKDMethod",
]


@register_method("gkd")
class GKDMethod:
    """GKD: Generalized Knowledge Distillation (On-Policy Distillation).

    Metadata class for the EasyOPD registry.
    """

    # Method metadata
    name = "gkd"
    description = (
        "GKD: Generalized Knowledge Distillation. "
        "On-policy distillation using generalized JSD with dense teacher feedback."
    )
    paper_url = "https://arxiv.org/abs/2306.13649"
    code_url = "https://github.com/shawnli/on-policy-distillation"

    # Files modified in verl
    verl_modified_files = [
        "verl/trainer/distillation/losses.py",   # Register GKD JSD loss mode
        "verl/workers/config/distillation.py",   # Add gkd_beta config field
    ]
