"""GAD configuration dataclass and the canonical `is_gad_enabled` check.

`is_gad_enabled(cfg)` is the SINGLE entry point used by every
`[EasyOPD:GAD]` if-branch in verl. Grepping for that string locates all
integration points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from omegaconf import DictConfig, OmegaConf


class GADConfigError(ValueError):
    """Raised when GAD is enabled but the surrounding config is invalid."""


def _select(node: Any, key: str, default: Any = None) -> Any:
    """OmegaConf.select with a default that works on omegaconf 2.0 and newer.

    The `default=` kwarg was added in 2.1; passing it explicitly to 2.0
    raises TypeError. We replicate the behavior locally.
    """
    if node is None:
        return default
    try:
        value = OmegaConf.select(node, key)
    except TypeError:
        return default
    return default if value is None else value


def is_gad_enabled(cfg: Any) -> bool:
    """Return True iff cfg.gad.enable is truthy.

    Defensive against missing `gad` node so verl runs without GAD config
    behave identically to before.
    """
    if cfg is None:
        return False
    gad_node = _select(cfg, "gad", default=None) if isinstance(cfg, DictConfig) else getattr(cfg, "gad", None)
    if gad_node is None:
        return False
    enable = _select(gad_node, "enable", default=False) if isinstance(gad_node, DictConfig) else getattr(gad_node, "enable", False)
    return bool(enable)


@dataclass(frozen=True)
class GADConfig:
    enable: bool = False
    discriminator_init_path: str | None = None
    metrics_prefix: str = "gad"

    @classmethod
    def load_from_omegaconf(cls, cfg: Any) -> "GADConfig":
        """Build and validate a GADConfig from the trainer's top-level cfg.

        Validation collects ALL violations and raises a single GADConfigError
        with a multi-line message, so the user can fix everything in one pass.
        """
        gad_node = _select(cfg, "gad", default=None) if isinstance(cfg, DictConfig) else getattr(cfg, "gad", None)
        enable = bool(_select(gad_node, "enable", default=False)) if gad_node is not None else False
        path = _select(gad_node, "discriminator_init_path", default=None) if gad_node is not None else None
        prefix = _select(gad_node, "metrics_prefix", default="gad") if gad_node is not None else "gad"

        if not enable:
            return cls(enable=False, discriminator_init_path=None, metrics_prefix=prefix or "gad")

        problems: List[str] = []

        if path in (None, "", "???"):
            problems.append(
                "gad.discriminator_init_path is required when gad.enable=true "
                f"(got {path!r})"
            )

        rm_enable = _select(cfg, "reward_model.enable", default=False)
        if bool(rm_enable):
            problems.append(
                "gad.enable=true is incompatible with reward_model.enable=true "
                "(GAD uses the critic as the reward source). Set reward_model.enable=false."
            )

        if problems:
            joined = "\n  - " + "\n  - ".join(problems)
            raise GADConfigError(f"GAD config has {len(problems)} problems:{joined}")

        return cls(enable=True, discriminator_init_path=str(path), metrics_prefix=prefix or "gad")
