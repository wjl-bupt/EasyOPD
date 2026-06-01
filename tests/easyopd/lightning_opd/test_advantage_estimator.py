"""Tests for on_policy_distillation advantage estimator."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import pytest


@pytest.fixture(autouse=True)
def _ensure_registered():
    from easyopd.methods.lightning_opd import register

    register()


def test_basic_advantage_computation():
    """advantage = teacher_lp - student_lp"""
    from easyopd.methods.lightning_opd.advantage_estimator import (
        compute_on_policy_distillation_advantages,
    )

    B, T = 2, 10
    student_lp = torch.randn(B, T)
    teacher_lp = student_lp + 0.5  # teacher is better
    response_mask = torch.ones(B, T)

    advantages, returns = compute_on_policy_distillation_advantages(
        token_level_rewards=student_lp,
        response_mask=response_mask,
        teacher_log_probs=teacher_lp,
    )

    expected = (teacher_lp - student_lp) * response_mask
    assert torch.allclose(advantages, expected)
    assert torch.allclose(returns, expected)


def test_ragged_batch():
    """Different response lengths per sample."""
    from easyopd.methods.lightning_opd.advantage_estimator import (
        compute_on_policy_distillation_advantages,
    )

    B, T = 3, 10
    student_lp = torch.randn(B, T)
    teacher_lp = torch.randn(B, T)
    # Mask: sample 0 has 5 tokens, sample 1 has 8, sample 2 has 10
    response_mask = torch.zeros(B, T)
    response_mask[0, :5] = 1
    response_mask[1, :8] = 1
    response_mask[2, :10] = 1

    advantages, returns = compute_on_policy_distillation_advantages(
        token_level_rewards=student_lp,
        response_mask=response_mask,
        teacher_log_probs=teacher_lp,
    )

    # Padded positions should be zero
    assert advantages[0, 5:].sum() == 0
    assert advantages[1, 8:].sum() == 0
    # Non-padded positions should be teacher - student
    assert torch.allclose(advantages[0, :5], (teacher_lp - student_lp)[0, :5])


def test_missing_teacher_logprobs_raises():
    """Should raise LightningOPDMissingTeacherLogprobs when teacher_log_probs is None."""
    from easyopd.methods.lightning_opd.advantage_estimator import (
        LightningOPDMissingTeacherLogprobs,
        compute_on_policy_distillation_advantages,
    )

    with pytest.raises(LightningOPDMissingTeacherLogprobs):
        compute_on_policy_distillation_advantages(
            token_level_rewards=torch.zeros(2, 10),
            response_mask=torch.ones(2, 10),
            teacher_log_probs=None,
        )


def test_register_name_resolvable():
    """on_policy_distillation should be resolvable after registration."""
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    fn = get_adv_estimator_fn("on_policy_distillation")
    assert callable(fn)
    assert fn.__name__ == "compute_on_policy_distillation_advantages"
