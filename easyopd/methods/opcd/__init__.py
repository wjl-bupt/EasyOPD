"""
OPCD: On-Policy Context Distillation for Language Models.

This method implements the OPCD framework which distills contextual knowledge
(experiential knowledge, system prompts) into a language model using on-policy
KL divergence minimization. The teacher model receives context in its prompt
while the student generates without context, then the student is trained to
match the teacher's output distribution.

Paper: https://arxiv.org/abs/2602.12275
Code: https://github.com/microsoft/LMOps/tree/main/opcd

Integration mode: Mode A (lightweight verl modification)
    - Core algorithm in easyopd/methods/opcd/core.py
    - Config fields added to verl/workers/config/actor.py (OPCDConfig)
    - KL loss computation added to verl/workers/actor/dp_actor.py
    - Consolidate stage logic added to verl/trainer/ppo/ray_trainer.py
    - Trainer config fields added to verl/trainer/config/ppo_trainer.yaml

Modified verl files:
    - verl/workers/config/actor.py: Added OPCDConfig with kl_loss_type, kl_topk, etc.
    - verl/workers/actor/dp_actor.py: Added OPCD KL loss computation branch
    - verl/trainer/ppo/ray_trainer.py: Added consolidate stage + experience injection
"""

from easyopd.methods.opcd.core import (
    kl_penalty,
    compute_opcd_loss,
    build_experience_prompt,
    truncate_experience,
    EXPERIENCE_SOLVE_PROMPT_TEMPLATE,
)
from easyopd.registry import register_method

__all__ = [
    "kl_penalty",
    "compute_opcd_loss",
    "build_experience_prompt",
    "truncate_experience",
    "EXPERIENCE_SOLVE_PROMPT_TEMPLATE",
    "OPCDMethod",
]


@register_method("opcd")
class OPCDMethod:
    """OPCD: On-Policy Context Distillation for Language Models.

    Metadata class for the EasyOPD registry.
    """

    # Method metadata
    name = "opcd"
    description = (
        "OPCD: On-Policy Context Distillation for Language Models. "
        "Distills contextual knowledge into LMs via on-policy KL minimization."
    )
    paper_url = "https://arxiv.org/abs/2602.12275"
    code_url = "https://github.com/microsoft/LMOps/tree/main/opcd"

    # Files modified in verl
    verl_modified_files = [
        "verl/workers/config/actor.py",          # Added OPCDConfig
        "verl/workers/actor/dp_actor.py",        # Added OPCD KL loss computation
        "verl/trainer/ppo/ray_trainer.py",       # Added consolidate stage logic
    ]
