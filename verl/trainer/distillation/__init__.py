"""verl.trainer.distillation - Distillation loss framework."""

from verl.trainer.distillation.losses import (
    DISTILLATION_LOSS_REGISTRY,
    DISTILLATION_SETTINGS_REGISTRY,
    DistillationLossSettings,
    register_distillation_loss,
    get_distillation_loss_fn,
    get_distillation_loss_settings,
    is_distillation_enabled,
)

__all__ = [
    "DISTILLATION_LOSS_REGISTRY",
    "DISTILLATION_SETTINGS_REGISTRY",
    "DistillationLossSettings",
    "register_distillation_loss",
    "get_distillation_loss_fn",
    "get_distillation_loss_settings",
    "is_distillation_enabled",
]
