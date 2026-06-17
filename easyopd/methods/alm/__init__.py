# Copyright 2026 EasyOPD Contributors
#
# alm: Approximate Likelihood Matching for cross-tokenizer knowledge
# distillation. Ported from KDFlow's `alm` algorithm into verl's on-policy
# distillation framework, sharing the EasyOPD `simple` teacher sidecar /
# teacher lm_head / overlap-vocab singleton infrastructure.
#
# Reference: "Universal Cross-Tokenizer Distillation via Approximate
# Likelihood Matching" (Minixhofer et al., NeurIPS 2025).

from dataclasses import dataclass

from easyopd.registry import register_method


@register_method("alm")
@dataclass(frozen=True)
class ALMMethod:
    """Static metadata describing the EasyOPD `alm` method."""

    name: str = "alm"
    legacy_name: str = "alm"
    loss_mode: str = "alm"
    verl_modified_files: tuple = (
        "verl/trainer/distillation/losses.py",
        "verl/workers/actor/dp_actor.py",
    )
    description: str = (
        "Approximate Likelihood Matching: align teacher/student token "
        "sequences into chunks via cumulative tokenizer.decode comparison, "
        "then minimize a binarised f-divergence (KL or TVD) between "
        "chunk-level log-probabilities. Cross-tokenizer compatible."
    )
    paper_url: str = "https://arxiv.org/abs/2503.20083"  # ALM (NeurIPS 2025)


METHOD = ALMMethod()


def register() -> None:
    """Trigger registration of the `alm` distillation loss into verl."""
    from .losses import register_alm_loss

    register_alm_loss()


__all__ = ["METHOD", "ALMMethod", "register"]
