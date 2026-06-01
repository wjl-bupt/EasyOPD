# Copyright 2026 EasyOPD Contributors
#
# simct: span-based cross-tokenizer knowledge distillation method, ported from
# KDFlow's `span_ctkd` algorithm and integrated into verl's on-policy
# distillation framework by reusing the EasyOPD `simple` teacher sidecar.

from dataclasses import dataclass

from easyopd.registry import register_method


@register_method("simct")
@dataclass(frozen=True)
class SimCTMethod:
    """Static metadata describing the EasyOPD `simct` method."""

    name: str = "simct"
    legacy_name: str = "span_ctkd"
    loss_mode: str = "simct"
    verl_modified_files: tuple = (
        "verl/trainer/distillation/losses.py",
        "verl/workers/config/distillation.py",
    )
    description: str = (
        "Span-based cross-tokenizer KD. This is the EasyOPD port of KDFlow's "
        "`span_ctkd`, using span virtual-vocabulary logits on top of the "
        "shared overlap vocabulary."
    )
    paper_url: str = "https://arxiv.org/abs/2410.XXXXX"  # SimCT paper


METHOD = SimCTMethod()


def register() -> None:
    """Trigger registration of the `simct` distillation loss into verl."""
    from .losses import register_simct_loss

    register_simct_loss()


__all__ = ["METHOD", "SimCTMethod", "register"]
