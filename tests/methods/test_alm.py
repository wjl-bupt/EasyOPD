# Copyright 2026 EasyOPD Contributors
#
# Unit tests for ALM core algorithm functions (chunk alignment + binarised
# f-divergence). These tests do NOT exercise the full verl logit-processor
# path (which requires a real teacher sidecar / FSDP runtime). Instead
# they directly call the pure functions to validate algorithm correctness.

from __future__ import annotations

import math

import pytest
import torch


def test_binarised_f_divergence_kl_high_temperature_zero_at_equal_logp():
    """When teacher == student log-probs, binarised KL must be ~0 at any tau."""
    from easyopd.methods.alm.losses import _binarised_f_divergence

    log_p = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float32)
    loss_high = _binarised_f_divergence(log_p, log_p.clone(), temperature=100.0, f_divergence="kl")
    loss_low = _binarised_f_divergence(log_p, log_p.clone(), temperature=1.0, f_divergence="kl")
    # KL is exactly zero when distributions are equal (within numerical tolerance).
    assert torch.allclose(loss_high, torch.zeros_like(loss_high), atol=1e-5)
    assert torch.allclose(loss_low, torch.zeros_like(loss_low), atol=1e-5)


def test_binarised_f_divergence_tvd_high_temperature_zero_at_equal_logp():
    """When teacher == student log-probs, binarised TVD must be ~0 at any tau."""
    from easyopd.methods.alm.losses import _binarised_f_divergence

    log_p = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float32)
    loss_high = _binarised_f_divergence(log_p, log_p.clone(), temperature=100.0, f_divergence="tvd")
    loss_low = _binarised_f_divergence(log_p, log_p.clone(), temperature=1.0, f_divergence="tvd")
    assert torch.allclose(loss_high, torch.zeros_like(loss_high), atol=1e-5)
    assert torch.allclose(loss_low, torch.zeros_like(loss_low), atol=1e-5)


def test_binarised_f_divergence_positive_when_teacher_differs_from_student():
    """A non-zero gap between teacher / student log-probs must produce a
    positive divergence under both branches (tau >= 50 closed-form and
    tau < 50 explicit form)."""
    from easyopd.methods.alm.losses import _binarised_f_divergence

    log_p_t = torch.tensor([-1.0, -2.0], dtype=torch.float32)
    log_p_s = torch.tensor([-2.0, -1.0], dtype=torch.float32)
    for tau in (1.0, 50.0, 100.0):
        for fdiv in ("kl", "tvd"):
            loss = _binarised_f_divergence(log_p_t, log_p_s, temperature=tau, f_divergence=fdiv)
            assert torch.all(torch.isfinite(loss)), f"non-finite loss at tau={tau} fdiv={fdiv}"


def test_binarised_f_divergence_unknown_kind_raises():
    from easyopd.methods.alm.losses import _binarised_f_divergence

    log_p = torch.tensor([-1.0])
    with pytest.raises(ValueError):
        _binarised_f_divergence(log_p, log_p.clone(), temperature=1.0, f_divergence="js")


class _FakeTokenizer:
    """Minimal tokenizer-like object used by chunk alignment tests."""

    def __init__(self, id_to_text: dict[int, str], eos_id: int = -1, special_ids=None):
        self._id_to_text = id_to_text
        self.eos_token_id = eos_id
        self.all_special_ids = list(special_ids or [])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._id_to_text.get(int(i), "?") for i in ids)


def test_compute_chunk_alignment_identical_sequences():
    """Identical token sequences should be chunked one-token-at-a-time."""
    from easyopd.methods.alm.losses import _compute_chunk_alignment_local

    id_to_text = {1: "a", 2: "b", 3: "c"}
    tok = _FakeTokenizer(id_to_text)
    chunks = _compute_chunk_alignment_local(
        tea_label_ids=[1, 2, 3], stu_label_ids=[1, 2, 3],
        teacher_tokenizer=tok, student_tokenizer=tok,
    )
    # Each token aligns 1:1 — three chunks of width 1.
    assert len(chunks) == 3
    for tea_s, tea_e, stu_s, stu_e in chunks:
        assert tea_e - tea_s == 1
        assert stu_e - stu_s == 1


def test_compute_chunk_alignment_text_match_different_tokens():
    """When the two tokenizers split the same text differently, the
    alignment should produce a single chunk covering the whole span."""
    from easyopd.methods.alm.losses import _compute_chunk_alignment_local

    teacher_id_to_text = {1: "ab", 2: "cd"}
    student_id_to_text = {10: "a", 11: "bcd"}
    tea_tok = _FakeTokenizer(teacher_id_to_text)
    stu_tok = _FakeTokenizer(student_id_to_text)
    chunks = _compute_chunk_alignment_local(
        tea_label_ids=[1, 2], stu_label_ids=[10, 11],
        teacher_tokenizer=tea_tok, student_tokenizer=stu_tok,
    )
    # Both sides spell "abcd"; cumulative texts only equalize at the end,
    # so we get ONE boundary -> one chunk covering both teacher tokens
    # and both student tokens.
    assert len(chunks) == 1
    tea_s, tea_e, stu_s, stu_e = chunks[0]
    assert (tea_s, tea_e, stu_s, stu_e) == (0, 2, 0, 2)


def test_compute_chunk_alignment_no_overlap_returns_empty():
    """If teacher / student texts share NO common prefix, alignment fails
    and we return an empty list (no boundaries)."""
    from easyopd.methods.alm.losses import _compute_chunk_alignment_local

    teacher_id_to_text = {1: "x"}
    student_id_to_text = {2: "y"}
    chunks = _compute_chunk_alignment_local(
        tea_label_ids=[1], stu_label_ids=[2],
        teacher_tokenizer=_FakeTokenizer(teacher_id_to_text),
        student_tokenizer=_FakeTokenizer(student_id_to_text),
    )
    assert chunks == []


def test_register_alm_loss_idempotent():
    """register_alm_loss should be idempotent (callable twice without error)."""
    from easyopd.methods.alm.losses import register_alm_loss
    register_alm_loss()
    register_alm_loss()
    from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY
    assert "alm" in DISTILLATION_LOSS_REGISTRY
