"""ROPD: Rubric-based On-policy Distillation.

Ported from the maintained ROPD mainline of black-opd into EasyOPD. The public
entrypoint is the `ropd` reward manager registered into verl's reward-manager
registry; runtime helpers live under `easyopd.methods.ropd.judge`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ROPDMethod:
    """Static metadata describing the EasyOPD `ropd` method."""

    name: str = "ropd"
    description: str = (
        "Rubric-based on-policy distillation. The reward manager evaluates each "
        "rollout against a teacher-grounded rubric using a teacher + rubricator "
        "+ verifier judge triple."
    )


METHOD = ROPDMethod()


def register() -> None:
    """Register the ROPD reward manager into verl's registry."""
    from easyopd.methods.ropd.reward_manager import register_ropd_reward_manager

    register_ropd_reward_manager()


__all__ = ["METHOD", "ROPDMethod", "register"]
