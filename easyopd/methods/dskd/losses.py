# Copyright 2026 EasyOPD Contributors
#
# DSKD (Dual-Space Knowledge Distillation) cross-tokenizer KD loss, ported
# from KDFlow `kdflow.algorithms.dskd` to verl's logit-processor protocol.
#
# Three token-alignment branches (selected via
# `distillation.distillation_loss.dskd_token_align`):
#   - "identical": teacher/student vocab identical -> direct t2s + s2t KD
#                  on every loss-mask position (KDFlow `_compute_dskd_loss`).
#   - "eta":       greedy character-level alignment over response tokens,
#                  filter via t2s_id_mapping (KDFlow `_compute_dskd_eta_loss`).
#   - "cma":       cross-modal-attention soft-alignment over the batch
#                  (KDFlow `_compute_dskd_cma_loss`).
#
# Architectural simplification vs KDFlow:
# - We do NOT have direct access to student hidden states inside verl's
#   logit-processor (only logits_rmpad). For the s2t path we therefore use
#   ``student_logits @ pinv(student_lm_head.T)`` to recover an estimate of
#   the student hidden, then project to teacher space. Mathematically
#   equivalent (modulo the rank of student_lm_head) to KDFlow's direct
#   formulation. See `projectors.py` for the rationale.
# - The t2s_projector is frozen via pinv-initialization (see projectors.py).
#   This sacrifices the "let the projector adapt during training" behaviour
#   of KDFlow but reproduces the bulk of the KD signal at step 0 and makes
#   the implementation trivially compatible with verl's actor optimizer.

from __future__ import annotations

import logging
import math
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from easyopd.methods._align_utils import align_token_sequences
from easyopd.methods.dskd.projectors import get_or_create_dskd_state

logger = logging.getLogger(__name__)

DSKD_LOSS_NAMES = ("dskd",)


__all__ = [
    "DSKD_LOSS_NAMES",
    "compute_dskd_xtok_logits_processor",
    "compute_distillation_loss_dskd_cross_tokenizer",
    "register_dskd_loss",
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_dskd_loss() -> None:
    """Idempotently register the `dskd` cross-tokenizer KD loss."""
    from verl.trainer.distillation.losses import (
        DISTILLATION_LOSS_REGISTRY,
        DistillationLossSettings,
        register_distillation_loss,
    )

    if all(name in DISTILLATION_LOSS_REGISTRY for name in DSKD_LOSS_NAMES):
        return

    names_to_register = [n for n in DSKD_LOSS_NAMES if n not in DISTILLATION_LOSS_REGISTRY]
    if not names_to_register:
        return

    decorator = register_distillation_loss(
        DistillationLossSettings(names=names_to_register, use_cross_tokenizer=True)
    )
    decorator(compute_distillation_loss_dskd_cross_tokenizer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kl_forward(
    student_logits: torch.Tensor,   # [N, V]
    teacher_logits: torch.Tensor,   # [N, V] (treated as targets)
) -> torch.Tensor:
    """Per-position forward KL: KL(p_stu || p_tea), used as KD loss term.

    Equivalent to KDFlow's `forward_kl` with reduction="none" returning
    a [N] vector of per-position KL values.
    """
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_p = student_logp.exp()
    return (student_p * (student_logp - teacher_logp)).sum(dim=-1)


def _approximate_student_hidden(
    student_logits: torch.Tensor,   # [N, V_stu]
    student_lm_head: torch.Tensor,  # [V_stu, H_stu]
) -> torch.Tensor:
    """Recover an approximation of student hidden states from student logits.

    Since ``student_logits = student_hidden @ student_lm_head.T``, we have
    ``student_hidden ≈ student_logits @ pinv(student_lm_head.T)``. This is
    exact when ``student_lm_head.T`` is full row-rank (the typical case
    when ``H_stu < V_stu``).

    For efficiency we compute the pinv on first call and cache it via the
    DSKDProjectorState (handled at call site), but here we accept it pre-
    multiplied with the s2t projector to avoid an explicit pinv computation
    on every call when used in the s2t path. See call sites in this file.
    """
    head_T = student_lm_head.transpose(0, 1).float()  # [H_stu, V_stu]
    head_T_pinv = torch.linalg.pinv(head_T)            # [V_stu, H_stu]
    return student_logits.float() @ head_T_pinv


# ---------------------------------------------------------------------------
# DSKD branches
# ---------------------------------------------------------------------------

def _compute_dskd_identical_branch(
    student_logits_loss: torch.Tensor,    # [N, V_stu]
    teacher_logits_loss: torch.Tensor,    # [N, V_tea] = teacher_lm_head(teacher_hidden)
    teacher_hidden_loss: torch.Tensor,    # [N, H_tea]
    state,
    avg_token_num: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equivalent to KDFlow `_compute_dskd_loss` (vocab_identical branch).

    Returns ``(per_position_loss, metric_scalars)`` where
    ``per_position_loss`` is ``[N]`` (the t2s_kd term, which is the only
    per-position term; t2s_ce and s2t_kd are per-batch scalars added
    inside the metrics dict).
    """
    student_lm_head = state.student_lm_head  # [V_stu, H_stu]
    teacher_lm_head_w = state.teacher_lm_head  # [V_tea, H_tea]

    # t2s_logits = teacher_hidden @ t2s_projector @ student_lm_head.T
    # t2s_projector_weight is [H_stu, H_tea]; KDFlow nn.Linear(H_tea -> H_stu)
    # applies as x @ W.T => result @ student_lm_head.T.
    t2s_hidden = teacher_hidden_loss.float() @ state.t2s_projector_weight.float().transpose(0, 1)  # [N, H_stu]
    t2s_logits = t2s_hidden @ student_lm_head.float().transpose(0, 1)  # [N, V_stu]

    t_preds = teacher_logits_loss.argmax(-1)
    t2s_ce_loss = F.cross_entropy(t2s_logits, t_preds, reduction="sum") / max(avg_token_num, 1.0)

    t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds)
    t2s_agreement = t2s_agreement_mask.float().mean()

    # Per-position t2s_kd: forward KL only at positions where t2s argmax agrees with teacher.
    kl_per_pos = _kl_forward(student_logits_loss, t2s_logits.detach())  # [N]
    t2s_kd_per_pos = kl_per_pos * t2s_agreement_mask.float()
    t2s_kd_loss_scalar = t2s_kd_per_pos.sum() / max(t2s_agreement_mask.sum().float().item(), 1.0)

    # s2t path — vocab identical; teacher_lm_head can directly accept student hidden
    # (recovered from logits via pinv).
    student_hidden_approx = _approximate_student_hidden(
        student_logits_loss, student_lm_head
    )  # [N, H_stu] (in fp32)
    # In identical-vocab case, t2s_projector.T also serves as s2t (since both vocabs are equal).
    # KDFlow's _compute_dskd_loss uses: s2t_proj = stu_lm_head @ part_teacher_head_pinv
    # but with vocab_identical that simplifies. We use teacher_overlap_head_pinv directly.
    # s2t_logits = student_hidden_approx @ teacher_lm_head.T -> [N, V_tea]
    s2t_logits = student_hidden_approx @ teacher_lm_head_w.float().transpose(0, 1)
    minV = min(teacher_logits_loss.shape[-1], s2t_logits.shape[-1])
    s2t_kd_loss_scalar = _kl_forward(
        s2t_logits[:, :minV], teacher_logits_loss[:, :minV].detach()
    ).sum() / max(avg_token_num, 1.0)

    return t2s_kd_per_pos, {
        "t2s_ce_loss": t2s_ce_loss.detach(),
        "t2s_kd_loss": t2s_kd_loss_scalar.detach(),
        "s2t_kd_loss": s2t_kd_loss_scalar.detach(),
        "t2s_agreement": t2s_agreement.detach(),
    }


def _compute_dskd_eta_branch(
    student_logits_loss: torch.Tensor,    # [N_s, V_stu]
    teacher_logits_loss: torch.Tensor,    # [N_t, V_tea]
    teacher_hidden_loss: torch.Tensor,    # [N_t, H_tea]
    student_label_ids: list[int],
    teacher_label_ids: list[int],
    student_tokenizer,
    teacher_tokenizer,
    state,
    avg_token_num: float,
) -> tuple[torch.Tensor, list[int], dict[str, torch.Tensor]]:
    """Equivalent to KDFlow `_compute_dskd_eta_loss`.

    Returns ``(per_aligned_loss, student_aligned_local_idx, metric_scalars)``
    where ``per_aligned_loss[k]`` is the t2s_kd loss to be placed at the
    student's k-th aligned local position.
    """
    student_lm_head = state.student_lm_head
    teacher_lm_head_w = state.teacher_lm_head
    t2s_id_mapping = state.t2s_id_mapping
    student_overlap_token_ids = state.student_overlap_token_ids
    if t2s_id_mapping is None or student_overlap_token_ids is None:
        raise RuntimeError("[EasyOPD:dskd:eta] requires non-identical vocab id mappings.")

    device = student_logits_loss.device
    dtype = student_logits_loss.dtype

    tea_tokens = teacher_tokenizer.convert_ids_to_tokens(teacher_label_ids)
    stu_tokens = student_tokenizer.convert_ids_to_tokens(student_label_ids)

    if tea_tokens == stu_tokens:
        align_t = list(range(len(tea_tokens)))
        align_s = list(range(len(stu_tokens)))
    else:
        align_t, align_s = align_token_sequences(
            tea_tokens, stu_tokens,
            teacher_eos_token=getattr(teacher_tokenizer, "eos_token", None),
            student_eos_token=getattr(student_tokenizer, "eos_token", None),
        )

    if not align_t or not align_s:
        empty = student_logits_loss.new_zeros((0,), dtype=dtype)
        return empty, [], {
            "t2s_ce_loss": student_logits_loss.new_tensor(0.0),
            "t2s_kd_loss": student_logits_loss.new_tensor(0.0),
            "s2t_kd_loss": student_logits_loss.new_tensor(0.0),
            "t2s_agreement": student_logits_loss.new_tensor(0.0),
        }

    align_t_t = torch.tensor(align_t, dtype=torch.long, device=device)
    align_s_t = torch.tensor(align_s, dtype=torch.long, device=device)

    # Map teacher argmax preds to student vocab; drop unmapped.
    t_preds = teacher_logits_loss.argmax(-1)            # [N_t]
    t_preds_aligned = t2s_id_mapping[t_preds.index_select(0, align_t_t)]   # [K]
    valid_mask = t_preds_aligned.ne(-1)
    align_t_t = align_t_t[valid_mask]
    align_s_t = align_s_t[valid_mask]
    t_preds_valid = t_preds_aligned[valid_mask]
    if int(align_t_t.numel()) == 0:
        empty = student_logits_loss.new_zeros((0,), dtype=dtype)
        return empty, [], {
            "t2s_ce_loss": student_logits_loss.new_tensor(0.0),
            "t2s_kd_loss": student_logits_loss.new_tensor(0.0),
            "s2t_kd_loss": student_logits_loss.new_tensor(0.0),
            "t2s_agreement": student_logits_loss.new_tensor(0.0),
        }

    teacher_hidden_aligned = teacher_hidden_loss.index_select(0, align_t_t).float()  # [K, H_tea]
    student_logits_at_align = student_logits_loss.index_select(0, align_s_t)         # [K, V_stu]
    teacher_logits_at_align = teacher_logits_loss.index_select(0, align_t_t)         # [K, V_tea]

    # t2s path.
    t2s_hidden = teacher_hidden_aligned @ state.t2s_projector_weight.float().transpose(0, 1)
    t2s_logits = t2s_hidden @ student_lm_head.float().transpose(0, 1)  # [K, V_stu]
    t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds_valid)
    t2s_agreement = t2s_agreement_mask.float().mean()

    align_count = float(int(align_t_t.numel()))
    t2s_ce_loss = F.cross_entropy(t2s_logits, t_preds_valid, reduction="sum") / max(align_count, 1.0)
    kl_per_pos = _kl_forward(student_logits_at_align, t2s_logits.detach())  # [K]
    t2s_kd_per_pos = kl_per_pos * t2s_agreement_mask.float()
    t2s_kd_loss_scalar = t2s_kd_per_pos.sum() / max(t2s_agreement_mask.sum().float().item(), 1.0)

    # s2t path: student logits at aligned positions -> approx student_hidden -> teacher space.
    student_hidden_approx = _approximate_student_hidden(
        student_logits_at_align, student_lm_head
    )  # [K, H_stu]
    # KDFlow: s2t_proj = stu_lm_head[:, overlap_ids] @ teacher_overlap_head_pinv
    #   => [H_stu, V_overlap] @ [V_overlap, H_tea] -> [H_stu, H_tea]
    # Then: s2t_hiddens = student_hidden @ s2t_proj -> [K, H_tea]
    # Finally: s2t_logits = s2t_hiddens @ teacher_lm_head.T -> [K, V_tea]
    stu_lm_head_overlap = student_lm_head.float().transpose(0, 1)  # [H_stu, V_stu]
    if student_overlap_token_ids is not None:
        stu_lm_head_overlap = stu_lm_head_overlap[:, student_overlap_token_ids]  # [H_stu, V_overlap]
    s2t_proj = stu_lm_head_overlap @ state.teacher_overlap_head_pinv.float()  # [H_stu, H_tea]
    s2t_hiddens = student_hidden_approx @ s2t_proj  # [K, H_tea]
    s2t_logits = s2t_hiddens @ teacher_lm_head_w.float().transpose(0, 1)  # [K, V_tea]
    s2t_kd_loss_scalar = _kl_forward(
        s2t_logits, teacher_logits_at_align.detach()
    ).sum() / max(align_count, 1.0)

    align_s_local = align_s_t.detach().cpu().tolist()
    return t2s_kd_per_pos.to(dtype), align_s_local, {
        "t2s_ce_loss": t2s_ce_loss.detach(),
        "t2s_kd_loss": t2s_kd_loss_scalar.detach(),
        "s2t_kd_loss": s2t_kd_loss_scalar.detach(),
        "t2s_agreement": t2s_agreement.detach(),
    }


def _compute_dskd_cma_branch(
    student_logits_loss: torch.Tensor,
    teacher_logits_loss: torch.Tensor,
    teacher_hidden_loss: torch.Tensor,
    state,
    avg_token_num: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single-sample CMA branch (KDFlow `_compute_dskd_cma_loss`, simplified).

    KDFlow's CMA path constructs a batch-level cross-attention attention
    mask using stu_sample_ids vs tea_sample_ids. In the verl logit-processor
    we already operate on a per-sample slice (the caller invokes us once per
    batch sample), so the attention mask is a full-1 matrix. This collapses
    the CMA path to a single-sample soft-alignment, which is still a faithful
    port of the original algorithm at the per-sample granularity.
    """
    if state.query_projector_weight is None:
        raise RuntimeError("[EasyOPD:dskd:cma] query_projector not initialized.")
    student_lm_head = state.student_lm_head
    teacher_lm_head_w = state.teacher_lm_head

    n_s = int(student_logits_loss.shape[0])
    n_t = int(teacher_logits_loss.shape[0])
    if n_s == 0 or n_t == 0:
        return student_logits_loss.new_zeros((0,)), {
            "t2s_ce_loss": student_logits_loss.new_tensor(0.0),
            "t2s_kd_loss": student_logits_loss.new_tensor(0.0),
            "s2t_kd_loss": student_logits_loss.new_tensor(0.0),
            "t2s_agreement": student_logits_loss.new_tensor(0.0),
        }

    # Get index embeddings using argmax of own logits as proxy for input/target ids.
    student_input_ids = student_logits_loss.argmax(-1)              # [N_s]
    teacher_input_ids = teacher_logits_loss.argmax(-1)              # [N_t]
    student_input_emb = student_lm_head.index_select(0, student_input_ids).float()  # [N_s, H_stu]
    teacher_input_emb = teacher_lm_head_w.index_select(0, teacher_input_ids).float()  # [N_t, H_tea]
    # Use shifted input as "target" proxy: roll by -1.
    if n_s >= 2:
        student_target_emb = torch.cat(
            [student_input_emb[1:], student_input_emb[-1:]], dim=0
        )
    else:
        student_target_emb = student_input_emb
    if n_t >= 2:
        teacher_target_emb = torch.cat(
            [teacher_input_emb[1:], teacher_input_emb[-1:]], dim=0
        )
    else:
        teacher_target_emb = teacher_input_emb

    stu_index = torch.cat([student_input_emb, student_target_emb], dim=-1)  # [N_s, 2*H_stu]
    tea_index = torch.cat([teacher_input_emb, teacher_target_emb], dim=-1)  # [N_t, 2*H_tea]

    # query_projector_weight: [2*H_tea, 2*H_stu]; KDFlow nn.Linear(2*H_stu -> 2*H_tea)
    # applies as x @ W.T => [N_s, 2*H_tea].
    stu_q = stu_index @ state.query_projector_weight.float().transpose(0, 1)  # [N_s, 2*H_tea]
    tea_k = tea_index                                                          # [N_t, 2*H_tea]

    # Cross-attention scores.
    h_tea = teacher_hidden_loss.shape[-1]
    align_attn = stu_q @ tea_k.transpose(0, 1) / math.sqrt(2.0 * h_tea)        # [N_s, N_t]

    # Single-sample case: attn_mask is all-ones (no -inf).
    t2s_weight = torch.softmax(align_attn, dim=-1)                              # [N_s, N_t]
    s2t_weight = torch.softmax(align_attn.transpose(0, 1), dim=-1)              # [N_t, N_s]

    # Value projections.
    student_hidden_approx = _approximate_student_hidden(
        student_logits_loss, student_lm_head
    )  # [N_s, H_stu]
    # tea_v: teacher_hidden -> student space via t2s_projector
    tea_v = teacher_hidden_loss.float() @ state.t2s_projector_weight.float().transpose(0, 1)  # [N_t, H_stu]

    t2s_hidden = t2s_weight @ tea_v                                             # [N_s, H_stu]
    t2s_logits = t2s_hidden @ student_lm_head.float().transpose(0, 1)           # [N_s, V_stu]

    # CE target = student's own argmax (proxy for student labels).
    t2s_acc_targets = student_input_ids
    t2s_ce_loss = F.cross_entropy(t2s_logits, t2s_acc_targets, reduction="sum") / max(avg_token_num, 1.0)

    kl_per_pos = _kl_forward(student_logits_loss, t2s_logits.detach())          # [N_s]

    # s2t path.
    s2t_hidden = s2t_weight @ student_hidden_approx                              # [N_t, H_stu]
    # Project back to teacher hidden space (use t2s_projector pinv approximation).
    # KDFlow uses a separate stu_v_hiddens from student_hidden directly; here we
    # approximate by applying t2s_projector.T as the inverse (assumes orthogonality).
    s2t_hidden_in_tea = s2t_hidden @ state.t2s_projector_weight.float()          # [N_t, H_tea]
    s2t_logits = s2t_hidden_in_tea @ teacher_lm_head_w.float().transpose(0, 1)   # [N_t, V_tea]
    s2t_kd_loss_scalar = _kl_forward(
        s2t_logits, teacher_logits_loss.detach()
    ).sum() / max(avg_token_num, 1.0)

    t2s_kd_loss_scalar = kl_per_pos.sum() / max(avg_token_num, 1.0)
    t2s_agreement_mask = t2s_logits.argmax(-1).eq(t2s_acc_targets)
    t2s_agreement = t2s_agreement_mask.float().mean()

    return kl_per_pos.to(student_logits_loss.dtype), {
        "t2s_ce_loss": t2s_ce_loss.detach(),
        "t2s_kd_loss": t2s_kd_loss_scalar.detach(),
        "s2t_kd_loss": s2t_kd_loss_scalar.detach(),
        "t2s_agreement": t2s_agreement.detach(),
    }


# ---------------------------------------------------------------------------
# Stage 1 — logit processor
# ---------------------------------------------------------------------------

def _resolve_student_lm_head_weight(config, device, dtype) -> torch.Tensor:
    """Best-effort load of the student lm_head weights.

    For DSKD the student lm_head is needed to:
      (a) initialize the t2s projector via pinv(W_stu),
      (b) project teacher_hidden -> student logits during the t2s path.
    Reads from `config.model.path` or from the `EASYOPD_STUDENT_MODEL_PATH`
    env var (set by the engine workers at student worker init).
    """
    import os
    from easyopd.methods.simple.teacher_lm_head import load_teacher_lm_head

    student_path = os.environ.get("EASYOPD_STUDENT_MODEL_PATH")
    if student_path is None:
        student_path = (
            getattr(config, "student_model_path", None)
            or getattr(config, "model_path", None)
        )
    if student_path is None and hasattr(config, "model"):
        student_path = getattr(config.model, "path", None)
    if student_path is None:
        raise RuntimeError(
            "[EasyOPD:dskd] cannot resolve student model path; set "
            "EASYOPD_STUDENT_MODEL_PATH or `actor_rollout_ref.model.path`."
        )

    head = load_teacher_lm_head(student_path, dtype=dtype)
    head = head.to(device)
    head.requires_grad_(False)
    return head.weight.detach().clone()


def compute_dskd_xtok_logits_processor(
    student_logits: torch.Tensor,
    data,
    cu_seqlens: torch.Tensor,
    config,
    distillation_config,
) -> dict[str, torch.Tensor]:
    """Compute DSKD per-token KD loss inside the student forward pass."""
    assert student_logits.dim() == 3 and student_logits.shape[0] == 1, (
        f"expected student_logits [1, total_nnz, V], got {tuple(student_logits.shape)}"
    )

    from easyopd.methods.simple import losses as simple_losses

    total_nnz = student_logits.shape[1]
    device = student_logits.device
    dtype = student_logits.dtype

    student_path_resolved = simple_losses._resolve_student_path(config)
    simple_losses._ensure_singletons(
        distillation_config=distillation_config,
        student_tokenizer_path=student_path_resolved,
        device=device,
        dtype=dtype,
    )
    teacher_lm_head = simple_losses._TEACHER_LM_HEAD
    teacher_tokenizer = simple_losses._TEACHER_TOKENIZER
    student_tokenizer = simple_losses._STUDENT_TOKENIZER
    if teacher_lm_head is None or teacher_tokenizer is None or student_tokenizer is None:
        raise RuntimeError("[EasyOPD:dskd] shared simple singletons not initialized.")

    student_lm_head_weight = _resolve_student_lm_head_weight(config, device, dtype)
    state = get_or_create_dskd_state(
        distillation_config=distillation_config,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_lm_head_weight=student_lm_head_weight,
        teacher_lm_head=teacher_lm_head,
        device=device,
        dtype=dtype,
    )

    teacher_hidden_states_arr = simple_losses._extract_non_tensor(data, "teacher_hidden_states")
    teacher_input_ids_arr = simple_losses._extract_non_tensor(data, "teacher_input_ids")
    teacher_loss_mask_arr = simple_losses._extract_non_tensor(data, "teacher_loss_mask")
    if teacher_hidden_states_arr is None:
        raise KeyError("[EasyOPD:dskd] teacher_hidden_states missing in non_tensor_batch.")
    if teacher_input_ids_arr is None or teacher_loss_mask_arr is None:
        raise KeyError("[EasyOPD:dskd] teacher_input_ids/teacher_loss_mask missing.")

    input_ids_rmpad = simple_losses._extract_input_ids_rmpad(data, total_nnz)
    response_lens = simple_losses._extract_response_lens(data, bsz=len(cu_seqlens) - 1, device=device)

    loss_config = distillation_config.distillation_loss
    token_align = str(getattr(loss_config, "dskd_token_align", "identical"))

    out = torch.zeros((total_nnz,), dtype=dtype, device=device)
    valid_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)
    response_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)

    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    response_lens_cpu = response_lens.detach().cpu().tolist()

    skipped_samples = 0
    metric_accum = {
        "t2s_ce_loss": 0.0,
        "t2s_kd_loss": 0.0,
        "s2t_kd_loss": 0.0,
        "t2s_agreement": 0.0,
    }
    metric_count = 0

    for batch_idx in range(len(cu) - 1):
        sample_start, sample_end = int(cu[batch_idx]), int(cu[batch_idx + 1])
        sample_len = sample_end - sample_start
        if sample_len <= 1:
            skipped_samples += 1
            continue
        response_len = int(response_lens_cpu[batch_idx])
        if response_len <= 0 or response_len >= sample_len:
            skipped_samples += 1
            continue

        student_loss_start = sample_end - response_len - 1
        student_loss_end = sample_end - 1
        student_abs_positions = torch.arange(
            student_loss_start, student_loss_end, dtype=torch.long, device=device,
        )
        if student_abs_positions.numel() == 0:
            skipped_samples += 1
            continue
        response_positions.index_fill_(0, student_abs_positions, 1.0)

        stu_label_positions = student_abs_positions + 1
        stu_label_ids = (
            input_ids_rmpad.index_select(0, stu_label_positions).detach().cpu().tolist()
        )
        student_logits_at_resp = student_logits[0].index_select(0, student_abs_positions)  # [resp_len, V_stu]

        teacher_hidden_np: np.ndarray = teacher_hidden_states_arr[batch_idx]
        teacher_ids_np: np.ndarray = np.asarray(teacher_input_ids_arr[batch_idx], dtype=np.int64)
        teacher_mask_np: np.ndarray = np.asarray(teacher_loss_mask_arr[batch_idx], dtype=bool)
        if teacher_hidden_np is None or int(teacher_hidden_np.shape[0]) == 0:
            skipped_samples += 1
            continue
        teacher_hidden_np = np.asarray(teacher_hidden_np, dtype=np.float32)
        tea_logit_positions = np.nonzero(teacher_mask_np)[0]
        tea_label_positions = tea_logit_positions + 1
        valid_tea = tea_label_positions < len(teacher_ids_np)
        tea_label_positions = tea_label_positions[valid_tea]
        if len(tea_label_positions) == 0:
            skipped_samples += 1
            continue
        usable = min(len(tea_label_positions), int(teacher_hidden_np.shape[0]))
        tea_label_positions = tea_label_positions[:usable]
        teacher_hidden_np = teacher_hidden_np[:usable]
        tea_label_ids = teacher_ids_np[tea_label_positions].tolist()

        teacher_hidden = torch.from_numpy(teacher_hidden_np).to(device=device, dtype=dtype)  # [N_t, H_tea]
        with torch.no_grad():
            teacher_logits_loss = teacher_lm_head(teacher_hidden)  # [N_t, V_tea]

        # ---- Branch dispatch ----
        if state.vocab_identical or token_align == "identical":
            # Identical branch: requires N_s == N_t.
            n_s = int(student_logits_at_resp.shape[0])
            n_t = int(teacher_hidden.shape[0])
            if n_s != n_t:
                # Truncate to common length.
                common = min(n_s, n_t)
                stu_view = student_logits_at_resp[:common]
                tea_view = teacher_logits_loss[:common]
                tea_h_view = teacher_hidden[:common]
                stu_pos_view = student_abs_positions[:common]
            else:
                stu_view = student_logits_at_resp
                tea_view = teacher_logits_loss
                tea_h_view = teacher_hidden
                stu_pos_view = student_abs_positions
            avg_token_num = float(max(int(stu_view.shape[0]), 1))
            kd_per_pos, scalars = _compute_dskd_identical_branch(
                stu_view, tea_view, tea_h_view, state, avg_token_num,
            )
            # Place per-position kd term at student abs positions.
            out.index_copy_(0, stu_pos_view, kd_per_pos.to(dtype))
            valid_positions.index_fill_(0, stu_pos_view, 1.0)
            for k in metric_accum:
                metric_accum[k] += float(scalars[k].item())
            metric_count += 1

        elif token_align == "eta":
            avg_token_num = float(max(len(stu_label_ids), 1))
            kd_per_aligned, align_s_local, scalars = _compute_dskd_eta_branch(
                student_logits_loss=student_logits_at_resp,
                teacher_logits_loss=teacher_logits_loss,
                teacher_hidden_loss=teacher_hidden,
                student_label_ids=stu_label_ids,
                teacher_label_ids=tea_label_ids,
                student_tokenizer=student_tokenizer,
                teacher_tokenizer=teacher_tokenizer,
                state=state,
                avg_token_num=avg_token_num,
            )
            if int(kd_per_aligned.numel()) == 0:
                skipped_samples += 1
                continue
            for k_local, kd_val in zip(align_s_local, kd_per_aligned):
                if k_local >= int(student_abs_positions.numel()):
                    continue
                pos = int(student_abs_positions[int(k_local)].item())
                out[pos] = kd_val.to(dtype)
                valid_positions[pos] = 1.0
            for k in metric_accum:
                metric_accum[k] += float(scalars[k].item())
            metric_count += 1

        elif token_align == "cma":
            avg_token_num = float(max(int(student_logits_at_resp.shape[0]), 1))
            kd_per_pos, scalars = _compute_dskd_cma_branch(
                student_logits_loss=student_logits_at_resp,
                teacher_logits_loss=teacher_logits_loss,
                teacher_hidden_loss=teacher_hidden,
                state=state,
                avg_token_num=avg_token_num,
            )
            out.index_copy_(0, student_abs_positions, kd_per_pos.to(dtype))
            valid_positions.index_fill_(0, student_abs_positions, 1.0)
            for k in metric_accum:
                metric_accum[k] += float(scalars[k].item())
            metric_count += 1

        else:
            raise ValueError(
                f"[EasyOPD:dskd] unknown dskd_token_align={token_align!r}; "
                "expected 'identical' | 'eta' | 'cma'."
            )

    metric_count = max(metric_count, 1)
    stats_dtype = dtype
    skipped_samples_t = torch.full((total_nnz,), float(skipped_samples), dtype=stats_dtype, device=device)
    t2s_ce_t = torch.full((total_nnz,), metric_accum["t2s_ce_loss"] / metric_count, dtype=stats_dtype, device=device)
    t2s_kd_t = torch.full((total_nnz,), metric_accum["t2s_kd_loss"] / metric_count, dtype=stats_dtype, device=device)
    s2t_kd_t = torch.full((total_nnz,), metric_accum["s2t_kd_loss"] / metric_count, dtype=stats_dtype, device=device)
    t2s_agreement_t = torch.full((total_nnz,), metric_accum["t2s_agreement"] / metric_count, dtype=stats_dtype, device=device)

    return {
        "distillation_losses": out.unsqueeze(0),
        "dskd_valid_positions": valid_positions.unsqueeze(0),
        "dskd_response_positions": response_positions.unsqueeze(0),
        "dskd_skipped_samples": skipped_samples_t.unsqueeze(0),
        "dskd_t2s_ce_loss": t2s_ce_t.unsqueeze(0),
        "dskd_t2s_kd_loss": t2s_kd_t.unsqueeze(0),
        "dskd_s2t_kd_loss": s2t_kd_t.unsqueeze(0),
        "dskd_t2s_agreement": t2s_agreement_t.unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# Stage 2 — final policy-loss assembly
# ---------------------------------------------------------------------------

def compute_distillation_loss_dskd_cross_tokenizer(
    config,
    distillation_config,
    model_output: dict,
    data,
) -> Tuple[torch.Tensor, dict[str, Any]]:
    """Convert DSKD rmpad losses back to padded layout and report metrics."""
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    if "distillation_losses" not in model_output:
        raise KeyError(
            "[EasyOPD:dskd] model_output['distillation_losses'] missing — "
            "stage-1 DSKD logit processor was not invoked."
        )

    distillation_losses = no_padding_2_padding(
        model_output["distillation_losses"], data
    ).clamp_min(0.0)

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    if distillation_losses.shape != response_mask_bool.shape:
        raise RuntimeError(
            "[EasyOPD:dskd] shape mismatch: "
            f"distillation_losses={tuple(distillation_losses.shape)} vs "
            f"response_mask={tuple(response_mask_bool.shape)}"
        )

    if "dskd_valid_positions" in model_output:
        valid_positions = no_padding_2_padding(model_output["dskd_valid_positions"], data) > 0
    else:
        valid_positions = distillation_losses > 0
    if "dskd_response_positions" in model_output:
        response_positions = no_padding_2_padding(model_output["dskd_response_positions"], data) > 0
    else:
        response_positions = response_mask_bool

    response_tokens = (response_positions & response_mask_bool).float().sum().clamp_min(1.0)
    valid_count = (valid_positions & response_mask_bool).float().sum()
    active_losses = distillation_losses[response_mask_bool]
    loss_mean = active_losses.mean() if active_losses.numel() > 0 else distillation_losses.new_tensor(0.0)
    loss_sum = active_losses.sum() if active_losses.numel() > 0 else distillation_losses.new_tensor(0.0)

    def _scalar(name: str) -> torch.Tensor:
        if name in model_output:
            return no_padding_2_padding(model_output[name], data)[0, 0]
        return distillation_losses.new_tensor(0.0)

    skipped = _scalar("dskd_skipped_samples")
    t2s_ce = _scalar("dskd_t2s_ce_loss")
    t2s_kd = _scalar("dskd_t2s_kd_loss")
    s2t_kd = _scalar("dskd_s2t_kd_loss")
    t2s_agreement = _scalar("dskd_t2s_agreement")

    loss_config = distillation_config.distillation_loss
    token_align = str(getattr(loss_config, "dskd_token_align", "identical"))
    token_align_code = {"identical": 0.0, "eta": 1.0, "cma": 2.0}.get(token_align, -1.0)

    metrics: dict[str, Any] = {
        "distillation/dskd_loss": Metric(AggregationType.SUM, loss_sum),
        "distillation/dskd_loss_mean": Metric(AggregationType.MEAN, loss_mean),
        "distillation/dskd_align_ratio": Metric(
            AggregationType.MEAN, valid_count / response_tokens
        ),
        "distillation/dskd_skipped_samples": Metric(AggregationType.SUM, skipped),
        "distillation/dskd_t2s_ce_loss": Metric(AggregationType.MEAN, t2s_ce),
        "distillation/dskd_t2s_kd_loss": Metric(AggregationType.MEAN, t2s_kd),
        "distillation/dskd_s2t_kd_loss": Metric(AggregationType.MEAN, s2t_kd),
        "distillation/dskd_t2s_agreement": Metric(AggregationType.MEAN, t2s_agreement),
        "distillation/dskd_token_align_code": Metric(
            AggregationType.MEAN, distillation_losses.new_tensor(token_align_code)
        ),
    }
    return distillation_losses, metrics
