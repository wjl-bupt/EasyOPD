# Copyright 2026 EasyOPD Contributors
#
# dskd: Dual-Space Knowledge Distillation cross-tokenizer KD method, ported
# from KDFlow's `dskd` algorithm into verl's on-policy distillation framework
# via the EasyOPD `simple` teacher sidecar / lm_head singleton.
#
# Reference: "Dual-Space Knowledge Distillation for Large Language Models"
# (Zhang et al., 2024).

from dataclasses import dataclass

from easyopd.registry import register_method


@register_method("dskd")
@dataclass(frozen=True)
class DSKDMethod:
    """Static metadata describing the EasyOPD `dskd` method."""

    name: str = "dskd"
    legacy_name: str = "dskd"
    loss_mode: str = "dskd"
    verl_modified_files: tuple = (
        "verl/trainer/distillation/losses.py",
        "verl/workers/actor/dp_actor.py",
    )
    description: str = (
        "Dual-Space Knowledge Distillation: project teacher hidden states to "
        "student space (t2s) and student hiddens to teacher space (s2t) via "
        "pinv-initialized projectors, then minimize a combination of t2s "
        "KD/CE and s2t KD losses. Supports three token-alignment branches: "
        "identical (same vocab), eta (greedy text alignment), cma "
        "(cross-modal attention). Cross-tokenizer compatible."
    )
    paper_url: str = "https://arxiv.org/abs/2406.17328"  # DSKD


METHOD = DSKDMethod()


def register() -> None:
    """Trigger registration of the `dskd` distillation loss into verl."""
    from .losses import register_dskd_loss

    register_dskd_loss()


__all__ = ["METHOD", "DSKDMethod", "register"]
