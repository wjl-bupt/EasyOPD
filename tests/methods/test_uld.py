# Copyright 2026 EasyOPD Contributors
#
# Unit tests for ULD core algorithm functions: closed-form Wasserstein-1
# distance and the shared character-level token alignment utility.

from __future__ import annotations

import pytest
import torch


def test_wasserstein_1_zero_when_logits_equal_topk_branch():
    """top-k branch: identical logits => W1 = 0."""
    from easyopd.methods.uld.losses import _compute_wasserstein_1

    torch.manual_seed(0)
    logits = torch.randn(4, 50, dtype=torch.float32)
    w1 = _compute_wasserstein_1(logits, logits.clone(), temperature=1.0, top_k=10)
    assert torch.allclose(w1, torch.zeros_like(w1), atol=1e-5)


def test_wasserstein_1_zero_when_logits_equal_full_vocab_branch():
    """full-vocab branch (top_k <= 0): identical logits => W1 = 0."""
    from easyopd.methods.uld.losses import _compute_wasserstein_1

    torch.manual_seed(1)
    logits = torch.randn(3, 30, dtype=torch.float32)
    w1 = _compute_wasserstein_1(logits, logits.clone(), temperature=1.0, top_k=-1)
    assert torch.allclose(w1, torch.zeros_like(w1), atol=1e-5)


def test_wasserstein_1_positive_when_logits_differ():
    """Different logits should give a positive finite W1 in both branches."""
    from easyopd.methods.uld.losses import _compute_wasserstein_1

    torch.manual_seed(2)
    s = torch.randn(2, 40, dtype=torch.float32)
    t = torch.randn(2, 60, dtype=torch.float32)  # different vocab sizes (cross-tokenizer)
    for top_k in (-1, 16):
        w1 = _compute_wasserstein_1(s, t, temperature=1.0, top_k=top_k)
        assert w1.shape == (2,)
        assert torch.all(torch.isfinite(w1)), f"non-finite W1 at top_k={top_k}"
        assert torch.all(w1 >= 0)


def test_wasserstein_1_top_k_approximation_close_to_full():
    """top-k approximation with large k should approach full-vocab result."""
    from easyopd.methods.uld.losses import _compute_wasserstein_1

    torch.manual_seed(3)
    s = torch.randn(2, 50, dtype=torch.float32)
    t = torch.randn(2, 50, dtype=torch.float32)
    w1_full = _compute_wasserstein_1(s, t, temperature=1.0, top_k=-1)
    w1_topk = _compute_wasserstein_1(s, t, temperature=1.0, top_k=49)
    # With top_k=V-1 and a residual bin, the result should match full-vocab
    # to within a small tolerance (the residual collects the smallest prob
    # which is near-zero for a 50-class softmax of random gaussians).
    assert torch.allclose(w1_full, w1_topk, atol=1e-3)


def test_align_token_sequences_identical_returns_trivial_alignment():
    """Identical token lists -> identical [0..N) on both sides."""
    from easyopd.methods._align_utils import align_token_sequences

    seq = ["the", "quick", "brown"]
    t_align, s_align = align_token_sequences(seq, seq, teacher_eos_token=None, student_eos_token=None)
    assert t_align == [0, 1, 2]
    assert s_align == [0, 1, 2]


def test_align_token_sequences_with_prefix_normalization():
    """SentencePiece (`▁`) and BPE (`Ġ`) prefixes should be stripped before comparison."""
    from easyopd.methods._align_utils import align_token_sequences

    teacher_tokens = ["\u2581the", "\u2581cat"]
    student_tokens = ["\u0120the", "\u0120cat"]
    t_align, s_align = align_token_sequences(
        teacher_tokens, student_tokens, teacher_eos_token=None, student_eos_token=None,
    )
    # After normalization both sequences are ["the", "cat"] -> trivial alignment.
    assert t_align == [0, 1]
    assert s_align == [0, 1]


def test_align_token_sequences_eos_normalisation():
    """EOS literals on both sides are mapped to a common marker."""
    from easyopd.methods._align_utils import align_token_sequences, EOS_TOKEN_MARKER

    teacher_tokens = ["a", "<|endoftext|>"]
    student_tokens = ["a", "</s>"]
    t_align, s_align = align_token_sequences(
        teacher_tokens, student_tokens,
        teacher_eos_token="<|endoftext|>",
        student_eos_token="</s>",
    )
    # Both sequences should compare equal after eos normalization.
    assert t_align == [0, 1]
    assert s_align == [0, 1]


def test_register_uld_loss_idempotent():
    from easyopd.methods.uld.losses import register_uld_loss
    register_uld_loss()
    register_uld_loss()
    from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY
    assert "uld" in DISTILLATION_LOSS_REGISTRY
