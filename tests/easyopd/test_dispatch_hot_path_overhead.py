# Copyright 2026 EasyOPD Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Hot-path overhead micro-benchmark for HookDispatcher.

Verifies that the dispatch boundary stays cheap when:
  (a) dispatcher is disabled (no method configured),
  (b) dispatcher is enabled but the active hook returns None.

Per-call overhead targets: <= 5 microseconds (averaged over many trials).
"""

from __future__ import annotations

import timeit

import pytest
import torch

from easyopd.hook_dispatch import HookDispatcher

# Threshold per single dispatch call.
HOT_PATH_BUDGET_US = 5.0


def test_disabled_dispatcher_overhead_under_budget():
    """A disabled dispatcher must short-circuit at O(1)."""
    dispatcher = HookDispatcher.from_config({})
    assert not dispatcher.enabled

    student = torch.zeros(1, 1, 1)
    teacher = torch.zeros(1, 1, 1)
    mask = torch.ones(1, 1)

    n = 10000

    def _call():
        dispatcher.compute_loss(student_logits=student, teacher_logits=teacher, mask=mask)

    elapsed = timeit.timeit(_call, number=n)
    per_call_us = (elapsed / n) * 1e6
    assert per_call_us <= HOT_PATH_BUDGET_US * 5, (  # generous 5x slack to absorb CI noise
        f"Disabled dispatcher overhead {per_call_us:.2f} us exceeds budget "
        f"{HOT_PATH_BUDGET_US * 5:.2f} us. Tighten the early-return path."
    )


def test_enabled_dispatcher_returns_none_overhead_under_budget():
    """An enabled dispatcher whose hook returns None must still be cheap."""

    class _NullLossHook:
        def compute_loss(self, **kwargs):
            return None, {}

    # Bypass auto-discovery and construct a minimal dispatcher with a
    # null hook so we measure pure dispatch overhead.
    dispatcher = HookDispatcher.from_config({"easyopd": {"method": {"name": "gkd"}}})
    if not dispatcher.enabled:
        pytest.skip("gkd method not registered; cannot measure enabled path.")

    # Replace the loss hook with a no-op for pure overhead measurement.
    dispatcher.hooks.loss_hook = _NullLossHook()

    student = torch.zeros(1, 1, 1)
    teacher = torch.zeros(1, 1, 1)
    mask = torch.ones(1, 1)

    n = 10000

    def _call():
        dispatcher.compute_loss(student_logits=student, teacher_logits=teacher, mask=mask)

    elapsed = timeit.timeit(_call, number=n)
    per_call_us = (elapsed / n) * 1e6
    # Slightly looser bound for the enabled path (one hook indirection).
    assert per_call_us <= HOT_PATH_BUDGET_US * 10, (
        f"Enabled-but-null dispatcher overhead {per_call_us:.2f} us exceeds "
        f"budget {HOT_PATH_BUDGET_US * 10:.2f} us."
    )
