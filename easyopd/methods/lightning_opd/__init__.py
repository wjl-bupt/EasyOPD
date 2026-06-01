"""
Lightning-OPD: Offline On-Policy Distillation for Large Reasoning Models.

Paper: https://arxiv.org/abs/2604.13010
Source: https://github.com/NVIDIA-NeMo/Lightning-OPD

This method implements offline precomputation of teacher log-probabilities
so that no live teacher server is needed during OPD training, reducing
cost by 3.6-4.0x and enabling MoE models that OOM with standard OPD.

Integration mode: advantage estimator + data adapter + teacher consistency
    - Advantage estimator registered into verl ADV_ESTIMATOR_REGISTRY
    - Data adapter reads teacher_log_probs from parquet into batch
    - Teacher consistency check ensures SFT teacher == OPD teacher

Modified verl files (all changes wrapped in
``# [EasyOPD:lightning_opd] ... # [EasyOPD:lightning_opd] End`` markers):
    - verl/trainer/ppo/ray_trainer.py: teacher_log_probs non-tensor→tensor hook

Public surface:
    * LightningOPDMethod                             (method.py)
    * compute_on_policy_distillation_advantages      (advantage_estimator.py)
    * attach_teacher_log_probs                       (data_adapter.py)
    * check_teacher_consistency                      (teacher_consistency.py)
    * LightningOPDTeacherInconsistency               (teacher_consistency.py)
    * LightningOPDLogprobLengthMismatch              (data_adapter.py)
    * LightningOPDMissingTeacherLogprobs             (advantage_estimator.py)
"""

from dataclasses import dataclass

from easyopd.registry import register_method


@register_method("lightning_opd")
@dataclass(frozen=True)
class LightningOPDMethod:
    """Static metadata describing the EasyOPD ``lightning_opd`` method."""

    name: str = "lightning_opd"
    verl_modified_files: tuple = (
        "verl/trainer/ppo/ray_trainer.py",
    )
    description: str = (
        "Offline on-policy distillation with precomputed teacher log-probabilities. "
        "Eliminates live teacher dependency during training, reducing cost 3.6-4.0x."
    )
    paper_url: str = "https://arxiv.org/abs/2604.13010"
    code_url: str = "https://github.com/NVIDIA-NeMo/Lightning-OPD"
    # Integration capabilities surfaced for the EasyOPD framework registry.
    # Lightning-OPD is unique among the unified methods in that it integrates
    # via verl's ADV_ESTIMATOR_REGISTRY plus a data adapter that lifts
    # precomputed teacher log-probabilities from parquet (non-tensor batch)
    # into a padded tensor before advantage computation. Actor / critic /
    # reward-manager surfaces are unchanged.
    capabilities: tuple = ("advantage_estimator", "data_adapter")
    integration_mode: str = "advantage-estimator"


METHOD = LightningOPDMethod()


def register() -> None:
    """Trigger registration of Lightning-OPD extensions into verl."""
    # Import lazily so importing this package does not pull torch/verl
    # before they are needed.
    from .advantage_estimator import compute_on_policy_distillation_advantages  # noqa: F401
    from .data_adapter import register_data_adapter

    register_data_adapter()


def __getattr__(name: str):
    """Lazily expose the public surface without importing verl on package import."""
    if name in {"compute_on_policy_distillation_advantages", "LightningOPDMissingTeacherLogprobs"}:
        from .advantage_estimator import (
            LightningOPDMissingTeacherLogprobs,
            compute_on_policy_distillation_advantages,
        )

        mapping = {
            "compute_on_policy_distillation_advantages": compute_on_policy_distillation_advantages,
            "LightningOPDMissingTeacherLogprobs": LightningOPDMissingTeacherLogprobs,
        }
        return mapping[name]

    if name in {"attach_teacher_log_probs", "LightningOPDLogprobLengthMismatch"}:
        from .data_adapter import LightningOPDLogprobLengthMismatch, attach_teacher_log_probs

        mapping = {
            "attach_teacher_log_probs": attach_teacher_log_probs,
            "LightningOPDLogprobLengthMismatch": LightningOPDLogprobLengthMismatch,
        }
        return mapping[name]

    if name in {"check_teacher_consistency", "LightningOPDTeacherInconsistency"}:
        from .teacher_consistency import LightningOPDTeacherInconsistency, check_teacher_consistency

        mapping = {
            "check_teacher_consistency": check_teacher_consistency,
            "LightningOPDTeacherInconsistency": LightningOPDTeacherInconsistency,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "METHOD",
    "LightningOPDMethod",
    "register",
    "compute_on_policy_distillation_advantages",
    "attach_teacher_log_probs",
    "check_teacher_consistency",
    "LightningOPDTeacherInconsistency",
    "LightningOPDLogprobLengthMismatch",
    "LightningOPDMissingTeacherLogprobs",
]
