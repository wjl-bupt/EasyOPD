# EasyOPD: methods that extend verl's on-policy distillation framework.
#
# This package only adds new methods on top of verl. Verl-side files only
# receive minimal, comment-bracketed extension hooks (search for
# `# [EasyOPD:simple]` markers in verl/).

"""EasyOPD — Unified On-Policy Distillation Framework.

Public API::

    from easyopd import EasyOPD

    # Discover all available methods
    print(EasyOPD.list_methods())

    # Load a method by name with optional config
    method = EasyOPD.from_hparams("gkd", config_path="easyopd/config/gkd.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from easyopd.data_provider import DataProvider, resolve_data_in_config
from easyopd.registry import (
    MethodNotFoundError,
    auto_discover,
    ensure_discovered,
    get_all_methods,
    get_method,
    is_registered,
    list_methods,
    register_method,
)

__all__ = [
    "EasyOPD",
    "DataProvider",
    "MethodNotFoundError",
    "auto_discover",
    "get_method",
    "list_methods",
    "register_method",
    "resolve_data_in_config",
]

# Default config directory (relative to this package)
_CONFIG_DIR = Path(__file__).parent / "config"


class EasyOPD:
    """Unified entry point for the EasyOPD framework.

    Provides class methods for method discovery, configuration loading,
    and method instantiation.
    """

    @classmethod
    def from_hparams(
        cls,
        method_name: str,
        config_path: Optional[str] = None,
        auto_resolve_data: bool = True,
        data_cache_dir: Optional[str] = None,
        **overrides: Any,
    ) -> "MethodInstance":
        """Load and instantiate an OPD method by name.

        Args:
            method_name: Registered method name (e.g. "simple", "gkd", "sod").
            config_path: Optional path to a YAML config file. If not provided,
                         looks for a default config at
                         ``easyopd/config/{method_name}.yaml``.
            auto_resolve_data: If True, automatically download and convert
                datasets specified by HuggingFace name in the config.
                Set to False to skip data resolution (e.g., in unit tests).
            data_cache_dir: Override cache directory for downloaded datasets.
            **overrides: Key-value pairs to override config values.

        Returns:
            A MethodInstance wrapping the method metadata and its resolved config.

        Raises:
            MethodNotFoundError: If the method name is not registered.
            FileNotFoundError: If the specified config_path does not exist.
        """
        # Ensure all methods are discovered
        ensure_discovered()

        # Get the method class from registry
        method_cls = get_method(method_name)

        # Resolve config path
        if config_path is None:
            default_path = _CONFIG_DIR / f"{method_name}.yaml"
            if default_path.exists():
                config_path = str(default_path)

        # Load config
        config = cls._load_config(config_path, overrides)

        # Ensure method name is in config for data resolution
        config.setdefault("method", {})
        config["method"].setdefault("name", method_name)

        # Auto-resolve data (download HuggingFace datasets if needed)
        if auto_resolve_data:
            config = resolve_data_in_config(config, cache_dir=data_cache_dir)

        return MethodInstance(
            method_cls=method_cls,
            method_name=method_name,
            config=config,
        )

    @classmethod
    def list_methods(cls) -> list[str]:
        """Return all registered method names after auto-discovery."""
        ensure_discovered()
        return list_methods()

    @classmethod
    def get_method(cls, name: str) -> Any:
        """Get a method class by name after auto-discovery."""
        ensure_discovered()
        return get_method(name)

    @staticmethod
    def _load_config(
        config_path: Optional[str], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Load a YAML config file and apply overrides.

        Args:
            config_path: Path to YAML file, or None.
            overrides: Key-value overrides to apply on top of loaded config.

        Returns:
            Merged configuration dictionary.
        """
        config: dict[str, Any] = {}

        if config_path is not None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Config file not found: {config_path}"
                )
            try:
                import yaml

                with open(config_path, "r") as f:
                    loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    config = loaded
            except ImportError:
                # Fallback: try omegaconf
                try:
                    from omegaconf import OmegaConf

                    cfg = OmegaConf.load(config_path)
                    config = OmegaConf.to_container(cfg, resolve=True)
                except ImportError:
                    raise ImportError(
                        "Either 'pyyaml' or 'omegaconf' is required to load "
                        "config files. Install with: pip install pyyaml"
                    )

        # Apply overrides
        if overrides:
            config.update(overrides)

        return config


class MethodInstance:
    """A resolved method instance with its metadata and configuration.

    Attributes:
        method_cls: The registered method metadata class.
        method_name: The string name of the method.
        config: The resolved configuration dictionary.
    """

    def __init__(
        self,
        method_cls: Any,
        method_name: str,
        config: dict[str, Any],
    ) -> None:
        self.method_cls = method_cls
        self.method_name = method_name
        self.config = config

    @property
    def description(self) -> str:
        """Return the method's description string."""
        return getattr(self.method_cls, "description", "")

    @property
    def paper_url(self) -> Optional[str]:
        """Return the method's paper URL if available."""
        return getattr(self.method_cls, "paper_url", None)

    @property
    def verl_modified_files(self) -> list[str]:
        """Return list of verl files modified by this method."""
        files = getattr(self.method_cls, "verl_modified_files", [])
        if isinstance(files, tuple):
            return list(files)
        return files

    @property
    def train_files(self) -> list[str]:
        """Return resolved training data file paths."""
        return self.config.get("data", {}).get("train_files", [])

    @property
    def val_files(self) -> list[str]:
        """Return resolved validation data file paths."""
        return self.config.get("data", {}).get("val_files", [])

    def resolve_data(self, cache_dir: Optional[str] = None) -> "MethodInstance":
        """Manually trigger data resolution (download + convert).

        Useful when auto_resolve_data=False was passed to from_hparams.
        """
        self.config = resolve_data_in_config(self.config, cache_dir=cache_dir)
        return self

    def __repr__(self) -> str:
        return (
            f"MethodInstance(name={self.method_name!r}, "
            f"config_keys={list(self.config.keys())})"
        )

