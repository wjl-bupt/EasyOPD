# Copyright 2026 EasyOPD Contributors
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

"""
SDPO: Self-Distillation Policy Optimization (Reinforcement Learning via Self-Distillation)

This method augments on-policy RL with self-distillation from the model's own
high-reward trajectories. An EMA copy of the policy conditioned on rich feedback
(a correct demonstration from the rollout group and/or environment feedback)
acts as a *self-teacher*; its feedback-informed next-token distribution is
distilled back into the policy via logit-level top-K generalized JSD. No external
teacher or reward model is needed.

This is a faithful reimplementation of the lasgroup/SDPO reference:
    - the self-teacher is a SEPARATE module (initialised from the base/reference
      model) that is EMA-updated towards the policy each optimizer step
      (teacher_regularization="ema", teacher_update_rate=0.05);
    - samples without a usable self-teacher contribute zero gradient (no GRPO
      fallback);
    - token-level rollout-correction IS weights multiply the distillation loss.

Paper: "Reinforcement Learning via Self-Distillation"
       Hübotter et al., 2026
       https://arxiv.org/abs/2601.20802
Code:  https://github.com/lasgroup/SDPO

Modified verl files:
    - verl/workers/actor/dp_actor.py: SDPO loss branch + EMA teacher forward/update
    - verl/workers/fsdp_workers.py: build the EMA self-teacher module for SDPO
    - verl/trainer/ppo/ray_trainer.py: SDPO reprompt batch + rollout correction
    - verl/trainer/ppo/rollout_corr_helper.py: rollout-correction IS weights
    - verl/workers/config/actor.py: SelfDistillationConfig (shared; SDPO fields)
"""

from easyopd.methods.sdpo.core import (
    build_reprompt_text,
    build_sdpo_teacher_inputs,
    compute_ema_update,
    compute_sdpo_actor_loss,
    compute_sdpo_loss,
    compute_sdpo_self_distillation_loss,
    remove_thinking_from_text,
    select_demonstration,
    sequence_rewards,
)
from easyopd.registry import register_method

from .hooks import SDPOLossHook, SDPOTeacherSidecarHook  # noqa: F401

__all__ = [
    "build_reprompt_text",
    "build_sdpo_teacher_inputs",
    "compute_ema_update",
    "compute_sdpo_actor_loss",
    "compute_sdpo_loss",
    "compute_sdpo_self_distillation_loss",
    "remove_thinking_from_text",
    "select_demonstration",
    "sequence_rewards",
    "SDPOMethod",
]


@register_method("sdpo")
class SDPOMethod:
    """SDPO: Self-Distillation Policy Optimization (RL via Self-Distillation).

    Metadata class for the EasyOPD registry.
    """

    # Method metadata
    name = "sdpo"
    description = (
        "SDPO: Self-Distillation Policy Optimization. On-policy RL augmented "
        "with self-distillation from the model's own feedback-informed "
        "(reprompted) self-teacher."
    )
    paper_url = "https://arxiv.org/abs/2601.20802"
    code_url = "https://github.com/lasgroup/SDPO"

    # Which hook capabilities this method exposes.
    capabilities = ("loss", "teacher_sidecar")

    # Files modified in verl (comment-bracketed `# [EasyOPD:SDPO]` markers).
    verl_modified_files = [
        "verl/workers/actor/dp_actor.py",        # SDPO loss branch + EMA teacher forward/update
        "verl/workers/fsdp_workers.py",          # build the EMA self-teacher module for SDPO
        "verl/trainer/ppo/ray_trainer.py",       # SDPO reprompt batch + rollout correction
        "verl/trainer/ppo/rollout_corr_helper.py",  # rollout-correction IS weights
        "verl/workers/config/actor.py",          # SelfDistillationConfig (shared; SDPO fields)
    ]
