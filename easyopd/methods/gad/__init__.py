# Copyright 2026 EasyOPD Contributors
#
# gad: Generative Adversarial Distillation (arXiv:2511.10643).
# Hacks verl's critic into a Bradley-Terry discriminator that scores
# student vs teacher responses, with the standard PPO actor consuming
# the discriminator output as a token-level reward signal.
#
# Verl files this method touches (all changes are wrapped in
# `# ============ [EasyOPD:GAD] ... # ============ [EasyOPD:GAD] End ============`
# comment markers):
#   * verl/workers/critic/dp_critic.py
#       - _forward_micro_batch: accept `compute_teacher` kwarg, swap
#         input keys via remap_to_teacher, and reduce to last-token-only.
#       - update_critic: dispatch to easyopd.methods.gad.critic_update
#         when gad.enable=true.

from dataclasses import dataclass

from easyopd.registry import register_method


@register_method("gad")
@dataclass(frozen=True)
class GADMethod:
    """Static metadata describing the EasyOPD `gad` method."""

    name: str = "gad"
    verl_modified_files: tuple = ("verl/workers/critic/dp_critic.py",)
    paper_url: str = "https://arxiv.org/abs/2511.10643"
    description: str = (
        "Generative Adversarial Distillation: repurposes the PPO critic "
        "as a Bradley-Terry discriminator over student vs teacher "
        "responses; the discriminator's last-token output drives the "
        "standard PPO advantage / actor update."
    )
    # Integration capabilities surfaced for the EasyOPD framework registry.
    # GAD is unique among the unified methods in that it modifies the PPO
    # *critic* (turning it into a Bradley-Terry discriminator) instead of
    # the actor loss / reward manager / teacher sidecar. The actor side
    # uses verl's standard PPO loss unchanged.
    capabilities: tuple = ("critic",)
    integration_mode: str = "critic-as-discriminator"


METHOD = GADMethod()


def register() -> None:
    """Idempotent registration entry point.

    GAD has no global registry side-effect to perform (unlike the loss-
    mode registration done by `simple`/`simct`): dispatch is config-
    driven via `is_gad_enabled(cfg)` checks in `verl/workers/critic/
    dp_critic.py`. This function exists so the method follows the same
    public shape as the other EasyOPD methods, and may be wired into a
    future central registry without changing call sites.
    """
    return None


__all__ = ["METHOD", "GADMethod", "register"]
