# Copyright 2026 EasyOPD Contributors
#
# uld: Universal Logit Distillation cross-tokenizer KD method, ported from
# KDFlow's `uld` algorithm into verl's on-policy distillation framework via
# the EasyOPD `simple` teacher sidecar / lm_head singleton.
#
# Reference: "Towards Cross-Tokenizer Distillation: the Universal Logit
# Distillation Loss for LLMs" (Boizard et al., TMLR 2025).

from dataclasses import dataclass

from easyopd.registry import register_method


@register_method("uld")
@dataclass(frozen=True)
class ULDMethod:
    """Static metadata describing the EasyOPD `uld` method."""

    name: str = "uld"
    legacy_name: str = "uld"
    loss_mode: str = "uld"
    verl_modified_files: tuple = (
        "verl/trainer/distillation/losses.py",
        "verl/workers/actor/dp_actor.py",
    )
    description: str = (
        "Universal Logit Distillation: closed-form Wasserstein-1 distance "
        "between sorted teacher/student probability vectors at greedy "
        "character-level aligned response token positions. Default uses "
        "top-k approximation (top_k=1024) to avoid full-vocab sort."
    )
    paper_url: str = "https://arxiv.org/abs/2402.12030"  # ULD (TMLR 2025)


METHOD = ULDMethod()


def register() -> None:
    """Trigger registration of the `uld` distillation loss into verl."""
    from .losses import register_uld_loss

    register_uld_loss()


__all__ = ["METHOD", "ULDMethod", "register"]
