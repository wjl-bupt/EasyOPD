# Copyright 2026 EasyOPD Contributors
#
# simple: cross-tokenizer knowledge distillation method (ported from
# KDFlow's `simple_ctkd`) that plugs into verl's on-policy distillation
# framework with minimal verl-side modifications.
#
# Public surface:
#   * find_overlap_tokens / align_sequences          (alignment.py)
#   * SGLangEngineService / EngineConfig             (sglang_engine.py)
#   * TeacherActorConfig / TeacherRayActor           (teacher_actor.py)
#   * TeacherActorGroup                              (teacher_group.py)
#   * load_teacher_lm_head                           (teacher_lm_head.py)
#   * compute_distillation_loss_simple_cross_tokenizer / register_simple_loss
#                                                    (losses.py)
#
# Verl files this method touches (all changes are wrapped in
# `# [EasyOPD:simple] ... # [EasyOPD:simple] End` comment markers):
#   * verl/trainer/distillation/losses.py
#       - DistillationLossSettings: add `use_cross_tokenizer` field and
#         relax mutual-exclusivity check to "exactly one of three".
#       - File-tail import that triggers `register_simple_loss()`.
#   * verl/workers/config/distillation.py
#       - DistillationTeacherModelConfig.validate_and_prepare_for_distillation:
#         accept `use_cross_tokenizer` and skip topk / response_length=1
#         rewrites in cross-tokenizer mode.
#       - DistillationConfig.__post_init__: forward the flag.

from dataclasses import dataclass

from easyopd.registry import register_method


@register_method("simple")
@dataclass(frozen=True)
class SimpleMethod:
    """Static metadata describing the EasyOPD `simple` method."""

    name: str = "simple"
    loss_mode: str = "simple"
    verl_modified_files: tuple = (
        "verl/trainer/distillation/losses.py",
        "verl/workers/config/distillation.py",
    )
    description: str = (
        "Cross-tokenizer KD that computes KL on the overlap sub-vocabulary "
        "between student and teacher tokenizers, with character-level "
        "greedy sequence alignment."
    )
    paper_url: str = "https://arxiv.org/abs/2410.XXXXX"  # Simple cross-tokenizer OPD


METHOD = SimpleMethod()


def register() -> None:
    """Trigger registration of the `simple` distillation loss into verl."""
    # Import lazily so importing this package does not pull torch/verl
    # before they are needed.
    from .losses import register_simple_loss

    register_simple_loss()


__all__ = ["METHOD", "SimpleMethod", "register"]
