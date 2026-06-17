"""Distillation loss registry and utilities for EasyOPD cross-tokenizer KD.

This module provides the registry infrastructure for distillation loss functions
(separate from the policy loss registry in core_algos.py). Methods like `simple`
and `simct` register their cross-tokenizer KD losses here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Type alias for distillation loss functions
DistillationLossFn = Callable[..., Tuple[Tensor, Dict[str, Any]]]

# Global registries
DISTILLATION_LOSS_REGISTRY: Dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: Dict[str, "DistillationLossSettings"] = {}


@dataclass
class DistillationLossSettings:
    """Settings for a registered distillation loss function.

    Attributes:
        names: List of names this loss is registered under.
        use_cross_tokenizer: Whether this loss requires cross-tokenizer alignment.
        use_hidden_states: Whether this loss requires teacher hidden states.
        use_logprobs: Whether this loss requires teacher log-probabilities.
    """

    names: List[str] = field(default_factory=list)
    use_cross_tokenizer: bool = False
    use_hidden_states: bool = False
    use_logprobs: bool = True


def register_distillation_loss(
    settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given settings.

    Args:
        settings: DistillationLossSettings describing the loss.

    Returns:
        Decorator that registers the loss function.
    """

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in settings.names:
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = settings
        return func

    return decorator


def get_distillation_loss_fn(name: str) -> DistillationLossFn:
    """Get a registered distillation loss function by name.

    Args:
        name: The registered name of the loss function.

    Returns:
        The distillation loss function.

    Raises:
        ValueError: If the name is not registered.
    """
    if name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported distillation loss mode: {name}. "
            f"Registered modes: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[name]


def get_distillation_loss_settings(name: str) -> DistillationLossSettings:
    """Get the settings for a registered distillation loss.

    Args:
        name: The registered name.

    Returns:
        DistillationLossSettings for the loss.
    """
    if name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"No settings for distillation loss: {name}. "
            f"Registered: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[name]


def is_distillation_enabled(config: Any) -> bool:
    """Check if distillation is enabled in the config.

    Args:
        config: Training configuration (OmegaConf or dict).

    Returns:
        True if distillation is enabled.
    """
    try:
        from omegaconf import OmegaConf

        if hasattr(config, "_metadata"):
            enabled = OmegaConf.select(config, "distillation.enabled", default=False)
            return bool(enabled)
    except ImportError:
        pass

    if isinstance(config, dict):
        return bool(config.get("distillation", {}).get("enabled", False))

    return False


# ---------------------------------------------------------------------------
# Auto-register EasyOPD distillation losses
# ---------------------------------------------------------------------------

# Register simple (cross-tokenizer KD) loss
try:
    from easyopd.methods.simple.losses import register_simple_loss as _register_simple_loss
    _register_simple_loss()
except Exception as _easyopd_simple_err:
    logger.debug(
        "Could not register EasyOPD simple loss: %s", _easyopd_simple_err
    )

# Register simct (span cross-tokenizer KD) loss
try:
    from easyopd.methods.simct.losses import register_simct_loss as _register_simct_loss
    _register_simct_loss()
except Exception as _easyopd_simct_err:
    logger.debug(
        "Could not register EasyOPD simct loss: %s", _easyopd_simct_err
    )

# ============ [EasyOPD:alm] Approximate Likelihood Matching (KDFlow port) ============
try:
    from easyopd.methods.alm.losses import register_alm_loss as _register_alm_loss
    _register_alm_loss()
except Exception as _easyopd_alm_err:
    logger.debug(
        "Could not register EasyOPD alm loss: %s", _easyopd_alm_err
    )
# ============ [EasyOPD:alm] End ============

# ============ [EasyOPD:uld] Universal Logit Distillation (KDFlow port) ============
try:
    from easyopd.methods.uld.losses import register_uld_loss as _register_uld_loss
    _register_uld_loss()
except Exception as _easyopd_uld_err:
    logger.debug(
        "Could not register EasyOPD uld loss: %s", _easyopd_uld_err
    )
# ============ [EasyOPD:uld] End ============

# ============ [EasyOPD:dskd] Dual-Space Knowledge Distillation (KDFlow port) ============
try:
    from easyopd.methods.dskd.losses import register_dskd_loss as _register_dskd_loss
    _register_dskd_loss()
except Exception as _easyopd_dskd_err:
    logger.debug(
        "Could not register EasyOPD dskd loss: %s", _easyopd_dskd_err
    )
# ============ [EasyOPD:dskd] End ============
