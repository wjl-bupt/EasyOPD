# Copyright 2026 EasyOPD Contributors
#
# Unit tests for DSKD core algorithm functions: helper KL, identical-branch
# loss with synthetic projector state, and registration idempotence.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch


def test_kl_forward_zero_when_logits_equal():
    from easyopd.methods.dskd.losses import _kl_forward

    torch.manual_seed(0)
    logits = torch.randn(3, 20, dtype=torch.float32)
    kl = _kl_forward(logits, logits.clone())
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_kl_forward_positive_when_logits_differ():
    from easyopd.methods.dskd.losses import _kl_forward

    torch.manual_seed(1)
    s = torch.randn(2, 16)
    t = torch.randn(2, 16)
    kl = _kl_forward(s, t)
    assert torch.all(torch.isfinite(kl))
    assert torch.all(kl >= 0)


def test_approximate_student_hidden_recovers_hidden():
    """student_hidden @ W.T = logits => pinv recovery should round-trip."""
    from easyopd.methods.dskd.losses import _approximate_student_hidden

    torch.manual_seed(2)
    H, V = 8, 16
    student_lm_head = torch.randn(V, H, dtype=torch.float32)
    student_hidden = torch.randn(4, H, dtype=torch.float32)
    student_logits = student_hidden @ student_lm_head.transpose(0, 1)
    recovered = _approximate_student_hidden(student_logits, student_lm_head)
    # The recovery is exact only if W.T has full column rank (H <= V here).
    assert torch.allclose(recovered, student_hidden, atol=1e-3)


# ---------------------------------------------------------------------------
# Synthetic state for identical-branch tests
# ---------------------------------------------------------------------------

def _build_identical_state(H_stu: int = 8, H_tea: int = 8, V: int = 12, device="cpu", dtype=torch.float32):
    """Build a DSKDProjectorState by hand for the identical-vocab branch."""
    from easyopd.methods.dskd.projectors import DSKDProjectorState

    torch.manual_seed(123)
    student_lm_head = torch.randn(V, H_stu, dtype=dtype, device=device)
    teacher_lm_head_w = torch.randn(V, H_tea, dtype=dtype, device=device)

    # t2s_projector via pinv (closed-form initial guess).
    student_head_hv = student_lm_head.transpose(0, 1)
    teacher_head_hv = teacher_lm_head_w.transpose(0, 1)
    pinv = torch.linalg.pinv(student_head_hv.float())
    init_t2s = (teacher_head_hv.float() @ pinv).transpose(0, 1).to(dtype)
    pinv_tea = torch.linalg.pinv(teacher_head_hv.float()).to(dtype)

    return DSKDProjectorState(
        student_lm_head=student_lm_head,
        teacher_lm_head=teacher_lm_head_w,
        t2s_projector_weight=init_t2s,
        teacher_overlap_head_pinv=pinv_tea,
        t2s_id_mapping=None,
        s2t_id_mapping=None,
        student_overlap_token_ids=None,
        teacher_overlap_token_ids=None,
        query_projector_weight=None,
        vocab_identical=True,
        device=torch.device(device),
        dtype=dtype,
    )


def test_dskd_identical_branch_returns_finite():
    from easyopd.methods.dskd.losses import _compute_dskd_identical_branch

    state = _build_identical_state()
    N = 4
    student_logits = torch.randn(N, 12, dtype=torch.float32)
    teacher_hidden = torch.randn(N, 8, dtype=torch.float32)
    teacher_logits = teacher_hidden @ state.teacher_lm_head.transpose(0, 1)

    kd_per_pos, scalars = _compute_dskd_identical_branch(
        student_logits_loss=student_logits,
        teacher_logits_loss=teacher_logits,
        teacher_hidden_loss=teacher_hidden,
        state=state,
        avg_token_num=float(N),
    )
    assert kd_per_pos.shape == (N,)
    assert torch.all(torch.isfinite(kd_per_pos))
    for k in ("t2s_ce_loss", "t2s_kd_loss", "s2t_kd_loss", "t2s_agreement"):
        assert k in scalars
        assert torch.isfinite(scalars[k]).all()


def test_dskd_cma_branch_runs_single_sample():
    from easyopd.methods.dskd.losses import _compute_dskd_cma_branch

    state = _build_identical_state()
    # CMA path needs query_projector_weight; build a block-diagonal init.
    H_stu = state.student_lm_head.shape[1]
    H_tea = state.teacher_lm_head.shape[1]
    block = torch.zeros((2 * H_tea, 2 * H_stu), dtype=torch.float32)
    t2s_T = state.t2s_projector_weight.transpose(0, 1)
    block[:H_tea, :H_stu] = t2s_T
    block[H_tea:, H_stu:] = t2s_T
    state.query_projector_weight = block

    N_s, N_t = 3, 3
    student_logits = torch.randn(N_s, state.student_lm_head.shape[0])
    teacher_hidden = torch.randn(N_t, H_tea)
    teacher_logits = teacher_hidden @ state.teacher_lm_head.transpose(0, 1)

    kd_per_pos, scalars = _compute_dskd_cma_branch(
        student_logits_loss=student_logits,
        teacher_logits_loss=teacher_logits,
        teacher_hidden_loss=teacher_hidden,
        state=state,
        avg_token_num=float(N_s),
    )
    assert kd_per_pos.shape == (N_s,)
    assert torch.all(torch.isfinite(kd_per_pos))
    for k in ("t2s_ce_loss", "t2s_kd_loss", "s2t_kd_loss"):
        assert torch.isfinite(scalars[k]).all()


def test_dskd_eta_branch_runs_with_synthetic_alignment():
    from easyopd.methods.dskd.losses import _compute_dskd_eta_branch
    from easyopd.methods.dskd.projectors import DSKDProjectorState

    torch.manual_seed(7)
    H_stu, H_tea, V_stu, V_tea = 6, 6, 10, 10
    student_lm_head = torch.randn(V_stu, H_stu)
    teacher_lm_head_w = torch.randn(V_tea, H_tea)
    pinv = torch.linalg.pinv(student_lm_head.transpose(0, 1).float())
    init_t2s = (teacher_lm_head_w.transpose(0, 1).float() @ pinv).transpose(0, 1)
    pinv_tea = torch.linalg.pinv(teacher_lm_head_w.transpose(0, 1).float())
    # Identity-ish id mapping over the full vocab.
    t2s_id_mapping = torch.arange(V_tea, dtype=torch.long)
    s2t_id_mapping = torch.arange(V_stu, dtype=torch.long)
    student_overlap_token_ids = torch.arange(V_stu, dtype=torch.long)
    teacher_overlap_token_ids = torch.arange(V_tea, dtype=torch.long)

    state = DSKDProjectorState(
        student_lm_head=student_lm_head,
        teacher_lm_head=teacher_lm_head_w,
        t2s_projector_weight=init_t2s,
        teacher_overlap_head_pinv=pinv_tea,
        t2s_id_mapping=t2s_id_mapping,
        s2t_id_mapping=s2t_id_mapping,
        student_overlap_token_ids=student_overlap_token_ids,
        teacher_overlap_token_ids=teacher_overlap_token_ids,
        query_projector_weight=None,
        vocab_identical=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    class _DummyTokenizer:
        eos_token = "<eos>"
        all_special_ids: list[int] = []

        def convert_ids_to_tokens(self, ids):
            return [f"t{int(i)}" for i in ids]

    tea_tok = _DummyTokenizer()
    stu_tok = _DummyTokenizer()

    student_logits = torch.randn(4, V_stu)
    teacher_hidden = torch.randn(4, H_tea)
    teacher_logits = teacher_hidden @ teacher_lm_head_w.transpose(0, 1)

    kd_per_aligned, align_s_local, scalars = _compute_dskd_eta_branch(
        student_logits_loss=student_logits,
        teacher_logits_loss=teacher_logits,
        teacher_hidden_loss=teacher_hidden,
        student_label_ids=[0, 1, 2, 3],
        teacher_label_ids=[0, 1, 2, 3],
        student_tokenizer=stu_tok,
        teacher_tokenizer=tea_tok,
        state=state,
        avg_token_num=4.0,
    )
    # With identical token strings, the trivial alignment should be returned.
    assert int(kd_per_aligned.numel()) == 4
    assert align_s_local == [0, 1, 2, 3]
    assert torch.all(torch.isfinite(kd_per_aligned))
    for k in ("t2s_ce_loss", "t2s_kd_loss", "s2t_kd_loss", "t2s_agreement"):
        assert torch.isfinite(scalars[k]).all()


def test_register_dskd_loss_idempotent():
    from easyopd.methods.dskd.losses import register_dskd_loss
    register_dskd_loss()
    register_dskd_loss()
    from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY
    assert "dskd" in DISTILLATION_LOSS_REGISTRY
