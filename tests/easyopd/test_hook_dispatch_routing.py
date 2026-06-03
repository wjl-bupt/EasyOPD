# Copyright 2026 EasyOPD Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Routing tests for HookDispatcher across all 12 EasyOPD methods.

Asserts that:
  - Each method registers correctly via @register_method.
  - HookDispatcher.from_config({...method.name=<m>...}) returns enabled=True.
  - dispatcher.hooks.active_hooks() contains at least one hook slot.
"""

from __future__ import annotations

import pytest

from easyopd.hook_dispatch import HookDispatcher
from easyopd.registry import (
    auto_discover,
    list_methods,
    resolve_method_name,
)

# The 12 canonical method names that must be supported.
ALL_METHODS = [
    "vision_opd",
    "opsa",
    "opcd",
    "sod",
    "g_opd",
    "ropd",
    "gad",
    "gkd",
    "sdpo",
    "lightning_opd",
    "simple",
    "simct",
]


@pytest.fixture(scope="module", autouse=True)
def _ensure_methods_discovered():
    """Trigger registry auto-discovery once for the whole module."""
    auto_discover()
    yield


@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_method_is_registered(method_name: str):
    """Every documented method must be registered."""
    registered = list_methods()
    assert method_name in registered, (
        f"Method '{method_name}' is missing from the registry. "
        f"Registered: {registered}"
    )


@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_dispatcher_routes_to_method(method_name: str):
    """HookDispatcher.from_config(...) routes to the correct method."""
    config = {"easyopd": {"method": {"name": method_name}}}
    dispatcher = HookDispatcher.from_config(config)
    assert dispatcher.enabled, (
        f"Dispatcher should be enabled for method '{method_name}', "
        f"got enabled={dispatcher.enabled}"
    )
    # gad (critic-side) and lightning_opd (dataloader-side) intentionally
    # do not register any of the 5 standard hooks; they are wired through
    # other extension points (custom critic worker / dataloader). See
    # easyopd/docs/verl-touchpoints.md for the rationale.
    if method_name in {"gad", "lightning_opd"}:
        return
    active = dispatcher.hooks.active_hooks() if hasattr(dispatcher.hooks, "active_hooks") else []
    assert len(active) >= 1, (
        f"Method '{method_name}' has no active hooks. "
        f"At least one of LossHook/RolloutHook/RewardHook/AlignmentHook/TeacherSidecarHook is expected."
    )


def test_vopd_alias_resolves_to_vision_opd():
    """The historical 'vopd' alias must resolve to the canonical 'vision_opd'."""
    assert resolve_method_name("vopd") == "vision_opd"


def test_dispatcher_disabled_when_no_method_configured():
    """When no easyopd.method.name is set, dispatcher must be a no-op."""
    dispatcher = HookDispatcher.from_config({})
    assert not dispatcher.enabled, (
        "Dispatcher must be disabled when no method is configured."
    )


def test_dispatcher_compute_loss_returns_none_when_disabled():
    """A disabled dispatcher's compute_loss must return (None, {})."""
    import torch

    dispatcher = HookDispatcher.from_config({})
    loss, metrics = dispatcher.compute_loss(
        student_logits=torch.zeros(2, 4, 8),
        teacher_logits=torch.zeros(2, 4, 8),
        mask=torch.ones(2, 4),
    )
    assert loss is None
    assert metrics == {}
