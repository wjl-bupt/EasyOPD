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

"""EasyOPD Method Registry.

Provides a unified mechanism for discovering, registering, and instantiating
OPD methods. Each method module uses the ``@register_method`` decorator on its
metadata class to make itself available through the registry.

Usage::

    from easyopd.registry import register_method, get_method, list_methods

    @register_method("my_method")
    class MyMethod:
        name = "my_method"
        description = "My custom OPD method"
        ...

    # Later:
    method_cls = get_method("my_method")
    all_methods = list_methods()
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Any, Callable, Optional, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Type[Any]] = {}


class MethodNotFoundError(KeyError):
    """Raised when a requested method name is not found in the registry."""

    def __init__(self, name: str) -> None:
        available = sorted(_REGISTRY.keys())
        msg = (
            f"Method '{name}' is not registered in EasyOPD. "
            f"Available methods: {available}"
        )
        super().__init__(msg)
        self.name = name
        self.available_methods = available


# ---------------------------------------------------------------------------
# Registration decorator
# ---------------------------------------------------------------------------


def register_method(name: str) -> Callable[[Type], Type]:
    """Decorator that registers an OPD method class under the given name.

    Args:
        name: The unique string identifier for this method (e.g. "simple",
              "gkd", "sod"). This is the name users will specify in config
              files or ``from_hparams()`` calls.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If a method with the same name is already registered.

    Example::

        @register_method("gkd")
        class GKDMethod:
            name = "gkd"
            description = "Generalized Knowledge Distillation"
            ...
    """

    def decorator(cls: Type) -> Type:
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            raise ValueError(
                f"Cannot register method '{name}': already registered by "
                f"{existing.__module__}.{existing.__qualname__}. "
                f"Duplicate registration attempted by "
                f"{cls.__module__}.{cls.__qualname__}."
            )
        _REGISTRY[name] = cls
        logger.debug("Registered EasyOPD method: %s -> %s", name, cls.__qualname__)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_method(name: str) -> Type[Any]:
    """Retrieve a registered method class by name.

    Args:
        name: The registered method name.

    Returns:
        The method metadata class.

    Raises:
        MethodNotFoundError: If the method name is not in the registry.
    """
    if name not in _REGISTRY:
        raise MethodNotFoundError(name)
    return _REGISTRY[name]


def list_methods() -> list[str]:
    """Return a sorted list of all registered method names."""
    return sorted(_REGISTRY.keys())


def get_all_methods() -> dict[str, Type[Any]]:
    """Return a copy of the full registry dictionary."""
    return dict(_REGISTRY)


def is_registered(name: str) -> bool:
    """Check whether a method name is registered."""
    return name in _REGISTRY


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

_DISCOVERED = False


def auto_discover(methods_package: Optional[str] = None) -> list[str]:
    """Scan and import all method sub-packages to trigger their registration.

    This function imports each sub-package under ``easyopd.methods`` (or the
    specified package), which causes their ``@register_method`` decorators to
    execute and populate the global registry.

    Args:
        methods_package: Dotted package path to scan. Defaults to
                         ``"easyopd.methods"``.

    Returns:
        List of method names that were newly discovered in this call.
    """
    global _DISCOVERED

    if methods_package is None:
        methods_package = "easyopd.methods"

    before = set(_REGISTRY.keys())

    try:
        package = importlib.import_module(methods_package)
    except ImportError as e:
        logger.warning("Cannot import methods package '%s': %s", methods_package, e)
        return []

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        logger.warning("Package '%s' has no __path__; cannot discover sub-packages.", methods_package)
        return []

    for importer, modname, ispkg in pkgutil.iter_modules(package_path):
        if not ispkg:
            continue
        full_name = f"{methods_package}.{modname}"
        try:
            importlib.import_module(full_name)
            logger.debug("Auto-discovered method package: %s", full_name)
        except Exception as e:
            logger.warning(
                "Failed to import method package '%s': %s", full_name, e
            )

    _DISCOVERED = True
    newly_found = sorted(set(_REGISTRY.keys()) - before)
    if newly_found:
        logger.info("Auto-discovered %d EasyOPD methods: %s", len(newly_found), newly_found)
    return newly_found


def ensure_discovered() -> None:
    """Ensure auto_discover() has been called at least once."""
    global _DISCOVERED
    if not _DISCOVERED:
        auto_discover()


# ---------------------------------------------------------------------------
# Convenience: reset (for testing)
# ---------------------------------------------------------------------------


def _reset_registry() -> None:
    """Clear the registry. Intended for use in tests only.

    Also removes method sub-packages from sys.modules so that
    re-importing them will re-execute @register_method decorators.
    """
    import sys

    global _DISCOVERED
    _REGISTRY.clear()
    _DISCOVERED = False

    # Remove cached method modules so re-import triggers registration
    to_remove = [key for key in sys.modules if key.startswith("easyopd.methods.")]
    for key in to_remove:
        del sys.modules[key]
