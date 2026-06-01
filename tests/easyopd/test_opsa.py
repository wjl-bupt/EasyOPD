"""
Unit tests for OPSA (On-Policy Self-Distillation for Safety Alignment) core functions.

Tests run on CPU and verify correctness of:
    - Teacher Flip Rate (TFR) computation
    - Early window weight computation (linear, step, exponential decay)
    - Per-token forward KL divergence
    - Full OPSA loss with temperature, masking, and aggregation

Usage:
    pytest tests/easyopd/test_opsa.py -v
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import pytest
import torch
import torch.nn.functional as F

from easyopd.methods.opsa.core import (
    compute_teacher_flip_rate,
    compute_early_window_weights,
    compute_opsa_kl_loss,
    opsa_loss,
)


# ============ Fixtures ============

@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 64


@pytest.fixture
def vocab_size():
    return 128


@pytest.fixture
def response_mask(batch_size, seq_len):
    """Create a response mask where first few tokens are padding (0), rest are response (1)."""
    mask = torch.zeros(batch_size, seq_len)
    # Each sample has response starting at different positions
    for i in range(batch_size):
        start = i * 2 + 2  # Start positions: 2, 4, 6, 8
        mask[i, start:] = 1.0
    return mask


@pytest.fixture
def student_logits(batch_size, seq_len, vocab_size):
    """Random student logits."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, vocab_size)


@pytest.fixture
def teacher_logits(batch_size, seq_len, vocab_size):
    """Random teacher logits (slightly different from student)."""
    torch.manual_seed(123)
    return torch.randn(batch_size, seq_len, vocab_size)


# ============ Tests for compute_teacher_flip_rate ============

class TestComputeTeacherFlipRate:
    """Tests for Teacher Flip Rate (TFR) computation."""

    def test_perfect_flip_rate(self):
        """All unsafe base responses become safe with context -> TFR = 1.0."""
        # Base: all unsafe (0), With context: all safe (1)
        flags_with_context = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
        flags_without_context = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

        tfr = compute_teacher_flip_rate(flags_with_context, flags_without_context)
        assert torch.isclose(tfr, torch.tensor(1.0)), f"Expected TFR=1.0, got {tfr.item()}"

    def test_zero_flip_rate(self):
        """No unsafe responses become safe -> TFR = 0.0."""
        flags_with_context = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
        flags_without_context = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

        tfr = compute_teacher_flip_rate(flags_with_context, flags_without_context)
        assert torch.isclose(tfr, torch.tensor(0.0)), f"Expected TFR=0.0, got {tfr.item()}"

    def test_partial_flip_rate(self):
        """Some unsafe become safe -> TFR between 0 and 1."""
        # Base: 4 unsafe, 1 safe
        flags_without_context = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)
        # With context: 2 of the 4 unsafe become safe
        flags_with_context = torch.tensor([1, 1, 0, 0, 1], dtype=torch.long)

        tfr = compute_teacher_flip_rate(flags_with_context, flags_without_context)
        # 2 flipped out of 4 unsafe -> TFR = 0.5
        assert torch.isclose(tfr, torch.tensor(0.5)), f"Expected TFR=0.5, got {tfr.item()}"

    def test_all_safe_base(self):
        """No unsafe base responses -> TFR = 0.0 (denominator is 0)."""
        flags_with_context = torch.tensor([1, 1, 1, 1], dtype=torch.long)
        flags_without_context = torch.tensor([1, 1, 1, 1], dtype=torch.long)

        tfr = compute_teacher_flip_rate(flags_with_context, flags_without_context)
        assert torch.isclose(tfr, torch.tensor(0.0)), f"Expected TFR=0.0, got {tfr.item()}"

    def test_output_range(self):
        """TFR should always be in [0, 1]."""
        torch.manual_seed(0)
        for _ in range(10):
            flags_with = torch.randint(0, 2, (20,))
            flags_without = torch.randint(0, 2, (20,))
            tfr = compute_teacher_flip_rate(flags_with, flags_without)
            assert 0.0 <= tfr.item() <= 1.0, f"TFR out of range: {tfr.item()}"


# ============ Tests for compute_early_window_weights ============

class TestComputeEarlyWindowWeights:
    """Tests for early refusal-decision window weight computation."""

    def test_output_shape(self, response_mask):
        """Output shape should match response_mask shape."""
        weights = compute_early_window_weights(response_mask, window_size=16)
        assert weights.shape == response_mask.shape

    def test_step_decay(self):
        """Step decay: 1.0 in window, min_weight outside."""
        mask = torch.zeros(1, 20)
        mask[0, 2:] = 1.0  # Response starts at position 2

        weights = compute_early_window_weights(mask, window_size=5, decay_type="step", min_weight=0.2)

        # First 2 positions (padding) should be 0
        assert weights[0, :2].sum() == 0.0
        # Positions 2-6 (first 5 response tokens) should be 1.0
        assert torch.allclose(weights[0, 2:7], torch.ones(5))
        # Remaining positions should be min_weight * mask
        assert torch.allclose(weights[0, 7:], torch.full((13,), 0.2))

    def test_linear_decay(self):
        """Linear decay: monotonically decreasing beyond window."""
        mask = torch.ones(1, 50)

        weights = compute_early_window_weights(mask, window_size=10, decay_type="linear", min_weight=0.1)

        # In-window tokens should be 1.0
        assert torch.allclose(weights[0, :10], torch.ones(10))
        # Beyond window: should be monotonically decreasing
        beyond_window = weights[0, 10:]
        for i in range(len(beyond_window) - 1):
            assert beyond_window[i] >= beyond_window[i + 1] - 1e-6

    def test_exponential_decay(self):
        """Exponential decay: monotonically decreasing beyond window."""
        mask = torch.ones(1, 50)

        weights = compute_early_window_weights(mask, window_size=10, decay_type="exponential", min_weight=0.1)

        # In-window tokens should be 1.0
        assert torch.allclose(weights[0, :10], torch.ones(10))
        # Beyond window: should be monotonically decreasing
        beyond_window = weights[0, 10:]
        for i in range(len(beyond_window) - 1):
            assert beyond_window[i] >= beyond_window[i + 1] - 1e-6

    def test_zero_mask(self):
        """All-zero mask should produce all-zero weights."""
        mask = torch.zeros(2, 30)
        weights = compute_early_window_weights(mask, window_size=10)
        assert weights.sum() == 0.0

    def test_min_weight_respected(self):
        """Weights should never go below min_weight for response tokens."""
        mask = torch.ones(1, 100)
        min_weight = 0.05

        for decay_type in ["linear", "step", "exponential"]:
            weights = compute_early_window_weights(
                mask, window_size=10, decay_type=decay_type, min_weight=min_weight
            )
            # All non-zero weights should be >= min_weight
            nonzero_weights = weights[weights > 0]
            assert (nonzero_weights >= min_weight - 1e-6).all(), \
                f"decay_type={decay_type}: found weight below min_weight"

    def test_invalid_decay_type(self):
        """Invalid decay type should raise ValueError."""
        mask = torch.ones(1, 20)
        with pytest.raises(ValueError):
            compute_early_window_weights(mask, window_size=5, decay_type="invalid")


# ============ Tests for compute_opsa_kl_loss ============

class TestComputeOpsaKlLoss:
    """Tests for per-token forward KL divergence D_KL(p_T || p_S)."""

    def test_output_shape(self, batch_size, seq_len, vocab_size):
        """Output should be (batch_size, seq_len)."""
        torch.manual_seed(42)
        student_lp = F.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
        teacher_lp = F.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)

        kl = compute_opsa_kl_loss(student_lp, teacher_lp)
        assert kl.shape == (batch_size, seq_len)

    def test_identical_distributions(self, batch_size, seq_len, vocab_size):
        """KL divergence of identical distributions should be 0."""
        torch.manual_seed(42)
        log_probs = F.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)

        kl = compute_opsa_kl_loss(log_probs, log_probs)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_non_negative(self, batch_size, seq_len, vocab_size):
        """KL divergence should be non-negative."""
        torch.manual_seed(42)
        student_lp = F.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
        teacher_lp = F.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)

        kl = compute_opsa_kl_loss(student_lp, teacher_lp)
        assert (kl >= 0).all(), "KL divergence should be non-negative"

    def test_asymmetry(self, batch_size, seq_len, vocab_size):
        """Forward KL D(T||S) != Reverse KL D(S||T) in general."""
        torch.manual_seed(42)
        student_lp = F.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
        teacher_lp = F.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)

        kl_forward = compute_opsa_kl_loss(student_lp, teacher_lp)  # D(T||S)
        kl_reverse = compute_opsa_kl_loss(teacher_lp, student_lp)  # D(S||T)

        # They should generally not be equal
        assert not torch.allclose(kl_forward, kl_reverse, atol=1e-3)


# ============ Tests for opsa_loss ============

class TestOpsaLoss:
    """Tests for the full OPSA loss function."""

    def test_output_format(self, student_logits, teacher_logits, response_mask):
        """Should return (loss_tensor, metrics_dict)."""
        loss, metrics = opsa_loss(student_logits, teacher_logits, response_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert isinstance(metrics, dict)
        assert "opsa/loss" in metrics
        assert "opsa/kl_mean" in metrics

    def test_loss_non_negative(self, student_logits, teacher_logits, response_mask):
        """Loss should be non-negative (KL is non-negative)."""
        loss, _ = opsa_loss(student_logits, teacher_logits, response_mask)
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_identical_logits_zero_loss(self, student_logits, response_mask):
        """Identical student and teacher logits should give ~0 loss."""
        loss, metrics = opsa_loss(student_logits, student_logits, response_mask)
        assert loss.item() < 1e-5, f"Expected ~0 loss for identical logits, got {loss.item()}"

    def test_temperature_effect(self, student_logits, teacher_logits, response_mask):
        """Higher temperature should generally reduce loss (softer distributions)."""
        loss_t1, _ = opsa_loss(student_logits, teacher_logits, response_mask, temperature=1.0)
        loss_t5, _ = opsa_loss(student_logits, teacher_logits, response_mask, temperature=5.0)

        # Higher temperature makes distributions more uniform, reducing KL
        # This is a soft check - not always guaranteed for all inputs
        # but should hold for random logits
        assert loss_t5.item() <= loss_t1.item() + 0.1  # Allow small tolerance

    def test_window_weighting_effect(self, student_logits, teacher_logits, response_mask):
        """With window weighting, loss should differ from without."""
        loss_with_window, metrics_with = opsa_loss(
            student_logits, teacher_logits, response_mask,
            use_window_weighting=True, window_size=8
        )
        loss_without_window, metrics_without = opsa_loss(
            student_logits, teacher_logits, response_mask,
            use_window_weighting=False
        )

        # Losses should generally differ when window weighting is applied
        # (unless all tokens are in the window)
        assert loss_with_window.item() != loss_without_window.item()

    def test_aggregation_modes(self, student_logits, teacher_logits, response_mask):
        """All aggregation modes should produce valid scalar loss."""
        for mode in ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"]:
            loss, metrics = opsa_loss(
                student_logits, teacher_logits, response_mask,
                loss_agg_mode=mode
            )
            assert loss.dim() == 0, f"Mode {mode}: loss should be scalar"
            assert not torch.isnan(loss), f"Mode {mode}: loss is NaN"
            assert not torch.isinf(loss), f"Mode {mode}: loss is inf"

    def test_zero_mask(self, student_logits, teacher_logits):
        """All-zero mask should produce zero loss."""
        batch_size, seq_len, _ = student_logits.shape
        zero_mask = torch.zeros(batch_size, seq_len)

        loss, _ = opsa_loss(student_logits, teacher_logits, zero_mask)
        assert loss.item() == 0.0, f"Expected 0 loss with zero mask, got {loss.item()}"

    def test_metrics_keys(self, student_logits, teacher_logits, response_mask):
        """Metrics dict should contain expected keys."""
        _, metrics = opsa_loss(
            student_logits, teacher_logits, response_mask,
            use_window_weighting=True
        )

        expected_keys = [
            "opsa/loss", "opsa/kl_mean",
            "opsa/kl_in_window", "opsa/kl_outside_window",
            "opsa/temperature", "opsa/window_size",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: {key}"

    def test_gradient_flow(self, student_logits, teacher_logits, response_mask):
        """Loss should allow gradient flow to student logits."""
        student_logits_grad = student_logits.clone().requires_grad_(True)

        loss, _ = opsa_loss(student_logits_grad, teacher_logits, response_mask)
        loss.backward()

        assert student_logits_grad.grad is not None
        assert not torch.all(student_logits_grad.grad == 0)


# ============ Integration test ============

class TestOPSAIntegration:
    """Integration tests verifying the full OPSA pipeline."""

    def test_import_from_package(self):
        """Verify OPSA can be imported from easyopd.methods.opsa."""
        from easyopd.methods.opsa import (
            compute_teacher_flip_rate,
            compute_opsa_kl_loss,
            compute_early_window_weights,
            opsa_loss,
            OPSAMethod,
        )
        assert OPSAMethod.name == "opsa"
        assert "arxiv" in OPSAMethod.paper_url

    def test_method_metadata(self):
        """Verify OPSAMethod metadata is complete."""
        from easyopd.methods.opsa import OPSAMethod

        assert hasattr(OPSAMethod, "name")
        assert hasattr(OPSAMethod, "description")
        assert hasattr(OPSAMethod, "paper_url")
        assert hasattr(OPSAMethod, "code_url")
        assert hasattr(OPSAMethod, "verl_modified_files")
        assert len(OPSAMethod.verl_modified_files) > 0
