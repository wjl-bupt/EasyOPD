# Copyright 2026 EasyOPD Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EasyOPD Configuration Loader.

Provides unified configuration loading, validation, and merging for all
OPD methods. Supports YAML config files with method-specific defaults
and user overrides.

Usage::

    from easyopd.config_loader import load_method_config, validate_config

    config = load_method_config("easyopd/config/gkd.yaml")
    validate_config(config, method_name="gkd")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default config directory
_CONFIG_DIR = Path(__file__).parent / "config"

# Required fields for each method (method_name -> list of required keys)
_METHOD_REQUIRED_FIELDS: dict[str, list[str]] = {
    "simple": ["distillation"],
    "simct": ["distillation"],
    "gkd": ["distillation"],
    "sod": [],
    "g_opd": [],
    "opcd": [],
    "vision_opd": [],
    "sdpo": [],
}


def load_method_config(
    config_path: Optional[str] = None,
    method_name: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Load and merge a method configuration.

    Loading priority (later overrides earlier):
    1. Method default config (from easyopd/config/{method_name}.yaml)
    2. User-specified config file (config_path)
    3. Programmatic overrides (overrides dict)

    Args:
        config_path: Path to a YAML config file. If None and method_name is
                     provided, loads the default config for that method.
        method_name: Name of the OPD method. Used to find default config.
        overrides: Dictionary of key-value overrides to apply on top.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If config_path is specified but doesn't exist.
        ValueError: If neither config_path nor method_name is provided.
    """
    if config_path is None and method_name is None:
        raise ValueError("Either config_path or method_name must be provided.")

    config: dict[str, Any] = {}

    # Step 1: Load method defaults
    if method_name is not None:
        default_path = _CONFIG_DIR / f"{method_name}.yaml"
        if default_path.exists():
            config = _load_yaml(str(default_path))
            logger.debug("Loaded method defaults from: %s", default_path)

    # Step 2: Load user config (overrides defaults)
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        user_config = _load_yaml(config_path)
        config = _deep_merge(config, user_config)
        logger.debug("Loaded user config from: %s", config_path)

    # Step 3: Apply programmatic overrides
    if overrides:
        config = _deep_merge(config, overrides)

    # Inject method name if not present
    if method_name and "method" not in config:
        config["method"] = {"name": method_name}
    elif method_name and isinstance(config.get("method"), dict):
        config["method"].setdefault("name", method_name)

    return config


def validate_config(
    config: dict[str, Any],
    method_name: Optional[str] = None,
) -> list[str]:
    """Validate a method configuration.

    Checks for required fields and reports warnings for unknown fields.

    Args:
        config: Configuration dictionary to validate.
        method_name: Method name for method-specific validation.

    Returns:
        List of warning messages (empty if all is well).

    Raises:
        ValueError: If required fields are missing.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # Determine method name
    if method_name is None:
        method_cfg = config.get("method", {})
        if isinstance(method_cfg, dict):
            method_name = method_cfg.get("name")

    if method_name is None:
        errors.append("Cannot determine method name from config. "
                      "Set 'method.name' in config or pass method_name parameter.")
        raise ValueError("; ".join(errors))

    # Check required fields
    required = _METHOD_REQUIRED_FIELDS.get(method_name, [])
    for field in required:
        if field not in config:
            errors.append(f"Required field '{field}' is missing for method '{method_name}'.")

    if errors:
        raise ValueError(
            f"Configuration validation failed for method '{method_name}': "
            + "; ".join(errors)
        )

    # Check for common issues
    if "distillation" in config:
        dist_cfg = config["distillation"]
        if isinstance(dist_cfg, dict):
            if dist_cfg.get("enabled") is False:
                warnings.append(
                    "distillation.enabled is False but a distillation method is configured."
                )

    return warnings


def get_default_config_path(method_name: str) -> Optional[str]:
    """Get the default config file path for a method.

    Args:
        method_name: The method name.

    Returns:
        Path string if default config exists, None otherwise.
    """
    path = _CONFIG_DIR / f"{method_name}.yaml"
    if path.exists():
        return str(path)
    return None


def list_available_configs() -> list[str]:
    """List all available method config files.

    Returns:
        List of method names that have default configs.
    """
    configs = []
    if _CONFIG_DIR.exists():
        for f in sorted(_CONFIG_DIR.iterdir()):
            if f.suffix == ".yaml" and f.stem != "__pycache__":
                configs.append(f.stem)
    return configs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file and return as dict."""
    try:
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(path)
            return OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            raise ImportError(
                "Either 'pyyaml' or 'omegaconf' is required to load config files. "
                "Install with: pip install pyyaml"
            )


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary.
        override: Override dictionary (values here win).

    Returns:
        New merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
