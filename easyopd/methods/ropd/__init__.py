"""
ROPD: Rubric-based On-policy Distillation.

Paper: https://github.com/Peregrine123/ROPD_official
Code: https://github.com/Peregrine123/ROPD_official
Integration mode: Mode B (black-box reward-manager)
    - Independent rubric/judge pipeline in easyopd/methods/ropd/{pipeline,
      judge, reward_manager}.py
    - Plugged into verl through verl's reward-manager registry; no actor-loss
      modification is required.
    - Bridged into the EasyOPD HookDispatcher via thin adapters in
      easyopd/methods/ropd/hooks.py (placeholder LossHook + RewardHook
      delegating to the reward manager).

Modified verl files:
    - verl/workers/reward_manager/__init__.py: registers the `ropd` reward
      manager so it is selectable via reward_model.reward_manager=ropd.
"""

from easyopd.registry import register_method

__all__ = ["METHOD", "ROPDMethod", "register"]


@register_method("ropd")
class ROPDMethod:
    """ROPD: Rubric-based On-policy Distillation.

    Metadata class for the EasyOPD registry. ROPD is a black-box OPD method
    that supplies dense rewards through a rubric pipeline (teacher +
    rubricator + verifier judge triple) rather than by altering the actor
    loss; integration with verl therefore happens at the reward-manager
    layer.
    """

    name = "ropd"
    description = (
        "ROPD: Rubric-based On-policy Distillation. The reward manager scores "
        "every rollout against a teacher-grounded rubric using a teacher + "
        "rubricator + verifier judge triple, and feeds the scalar reward back "
        "into verl's standard PG training loop."
    )
    paper_url = "https://github.com/Peregrine123/ROPD_official"
    code_url = "https://github.com/Peregrine123/ROPD_official"
    integration_mode = "reward-manager"  # black-box: no actor-loss modification

    verl_modified_files = [
        # Registers the `ropd` reward manager so it is selectable via
        # reward_model.reward_manager=ropd in verl configs.
        "verl/workers/reward_manager/__init__.py",
    ]


# Module-level singleton instance, kept for backward compatibility with the
# original ROPD package layout that imported `METHOD` directly.
METHOD = ROPDMethod()


def register() -> None:
    """Register the ROPD reward manager into verl's registry.

    This is the legacy entrypoint preserved from the upstream ROPD mainline;
    it wires the `ropd` reward manager into verl so that
    ``reward_model.reward_manager=ropd`` resolves correctly. The new
    EasyOPD-level registry integration above (``@register_method("ropd")``)
    is independent of this call and does not replace it.
    """
    from easyopd.methods.ropd.reward_manager import register_ropd_reward_manager

    register_ropd_reward_manager()
