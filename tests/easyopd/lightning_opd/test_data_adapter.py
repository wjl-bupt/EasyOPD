"""Tests for data_adapter teacher_log_probs conversion."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import torch
import pytest


class FakeDataProto:
    """Minimal DataProto stub for testing."""

    def __init__(self, batch: dict, non_tensor_batch: dict):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch


def test_attach_teacher_log_probs_basic():
    """Ragged teacher_log_probs should be padded to max response length."""
    from easyopd.methods.lightning_opd.data_adapter import attach_teacher_log_probs

    B, T = 3, 20
    response_mask = torch.zeros(B, T)
    response_mask[0, :5] = 1
    response_mask[1, :8] = 1
    response_mask[2, :10] = 1

    teacher_lps = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ], dtype=object)

    batch = FakeDataProto(
        batch={"response_mask": response_mask},
        non_tensor_batch={"teacher_log_probs": teacher_lps},
    )

    attach_teacher_log_probs(batch)

    assert "teacher_log_probs" in batch.batch
    tensor = batch.batch["teacher_log_probs"]
    assert tensor.shape == (B, 10)  # max response length
    assert tensor.dtype == torch.float32
    assert torch.isclose(tensor[0, 0], torch.tensor(0.1))
    assert torch.isclose(tensor[0, 4], torch.tensor(0.5))
    # Padded positions should be -inf
    assert tensor[0, 5] == float("-inf")


def test_no_op_when_column_missing():
    """Should not modify batch when teacher_log_probs is absent."""
    from easyopd.methods.lightning_opd.data_adapter import attach_teacher_log_probs

    batch = FakeDataProto(
        batch={"response_mask": torch.ones(2, 10)},
        non_tensor_batch={},
    )

    attach_teacher_log_probs(batch)
    assert "teacher_log_probs" not in batch.batch


def test_no_op_when_no_non_tensor_batch():
    """Should handle missing non_tensor_batch gracefully."""
    from easyopd.methods.lightning_opd.data_adapter import attach_teacher_log_probs

    batch = FakeDataProto(
        batch={"response_mask": torch.ones(2, 10)},
        non_tensor_batch={},
    )
    # Simulate no non_tensor_batch attr
    delattr(batch, "non_tensor_batch")

    attach_teacher_log_probs(batch)
    assert "teacher_log_probs" not in batch.batch


def test_truncation_when_lp_too_long():
    """Should truncate teacher_log_probs when longer than response mask."""
    from easyopd.methods.lightning_opd.data_adapter import attach_teacher_log_probs

    B, T = 1, 10
    response_mask = torch.zeros(B, T)
    response_mask[0, :3] = 1  # only 3 response tokens

    teacher_lps = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=object)

    batch = FakeDataProto(
        batch={"response_mask": response_mask},
        non_tensor_batch={"teacher_log_probs": teacher_lps},
    )

    attach_teacher_log_probs(batch)
    tensor = batch.batch["teacher_log_probs"]
    assert tensor.shape == (1, 3)  # max response length = 3
    assert torch.isclose(tensor[0, 2], torch.tensor(0.3))
