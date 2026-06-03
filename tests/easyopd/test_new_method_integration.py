# Copyright 2026 EasyOPD Contributors

"""Integration test: a brand-new method `echo_kd` plugged in only under
`easyopd/methods/echo_kd/` should be fully usable, without any verl change.
"""

from __future__ import annotations

import pytest
import torch

from easyopd.hook_dispatch import HookDispatcher
from easyopd.registry import auto_discover, list_methods


@pytest.fixture(scope="module", autouse=True)
def _ensure_methods_discovered():
    auto_discover()
    yield


def test_echo_kd_is_registered():
    auto_discover()
    assert "echo_kd" in list_methods(), (
        "Demo method `echo_kd` must be auto-discovered under easyopd.methods."
    )


def test_echo_kd_dispatcher_enabled():
    cfg = {"easyopd": {"method": {"name": "echo_kd"}}}
    dispatcher = HookDispatcher.from_config(cfg)
    assert dispatcher.enabled, (
        "HookDispatcher must enable for the new echo_kd method."
    )
    assert dispatcher.hooks.has_loss


def test_echo_kd_compute_loss_returns_valid_loss():
    cfg = {"easyopd": {"method": {"name": "echo_kd"}}}
    dispatcher = HookDispatcher.from_config(cfg)

    bsz, seq_len, vocab = 2, 4, 8
    student = torch.randn(bsz, seq_len, vocab, requires_grad=True)
    teacher = torch.randn(bsz, seq_len, vocab)
    mask = torch.ones(bsz, seq_len)

    loss, metrics = dispatcher.compute_loss(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert "echo_kd/mean_sq_diff" in metrics
    # Sanity: loss is non-negative (MSE).
    assert float(loss.detach()) >= 0.0
