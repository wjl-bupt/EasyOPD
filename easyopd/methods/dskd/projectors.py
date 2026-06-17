# Copyright 2026 EasyOPD Contributors
#
# DSKD projector state — initialization, caching and (optional) gradient
# clamp hook. Equivalent to KDFlow `dskd._init_projectors` but adapted to
# verl's logit-processor architecture, where we do NOT have direct access
# to student hidden states inside the per-micro-batch processor.
#
# Architectural simplification vs KDFlow:
# - In KDFlow the projector is a trainable nn.Linear and its parameters
#   are added to the actor optimizer with a configurable lr scale. In
#   verl's FSDP / Ray pipeline injecting extra param groups into the actor
#   optimizer is non-trivial and would require deep verl-side changes. We
#   therefore initialize the projector via the closed-form pseudo-inverse
#   solution (which reproduces teacher logits exactly when teacher and
#   student lm_heads are well-conditioned) and FREEZE it as a buffer.
#   This is mathematically equivalent to KDFlow's pre-training initial
#   projector and produces the bulk of the KD signal: t2s_logits will
#   approximately match teacher_logits at initialisation, after which
#   the student lm_head being updated through KD is what closes the gap.
# - The optional gradient-clamp hook is therefore only relevant if a
#   future revision re-enables trainable projectors. We keep the API
#   surface (`_grad_clamp_hook`) so that path can be re-enabled later
#   without changing call sites.

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


__all__ = ["DSKDProjectorState", "get_or_create_dskd_state"]


@dataclass
class DSKDProjectorState:
    """Process-level cache of the DSKD projector state."""

    # Frozen tensors used by both identical / cma / eta branches.
    student_lm_head: torch.Tensor          # [V_stu, H_stu], detached
    teacher_lm_head: torch.Tensor          # [V_tea, H_tea], detached (taken from teacher singleton)
    t2s_projector_weight: torch.Tensor     # [H_stu, H_tea], pinv-init, frozen
    teacher_overlap_head_pinv: torch.Tensor  # [H_tea, K], for s2t path; K = min(overlap, topk)

    # Cross-tokenizer mapping tensors (only populated if vocab_identical=False).
    t2s_id_mapping: Optional[torch.Tensor] = None    # [V_tea] long, -1 = unmapped
    s2t_id_mapping: Optional[torch.Tensor] = None    # [V_stu] long, -1 = unmapped
    student_overlap_token_ids: Optional[torch.Tensor] = None  # [K_overlap] long
    teacher_overlap_token_ids: Optional[torch.Tensor] = None  # [K_overlap] long

    # CMA-specific projector (also frozen in this implementation).
    query_projector_weight: Optional[torch.Tensor] = None     # [2*H_tea, 2*H_stu]

    vocab_identical: bool = False
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None


_LOCK = threading.Lock()
_STATE: Optional[DSKDProjectorState] = None


def _build_id_mappings(
    student_tokenizer,
    teacher_tokenizer,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Build teacher_id <-> student_id mappings via normalized vocab intersection.

    Equivalent to KDFlow `dskd._init_vocab_mapping`. Both maps default to
    -1 (unmapped); special tokens (eos/bos/pad) are explicitly forced.
    """
    student_vocab = {k.replace("\u0120", "\u2581"): v for k, v in student_tokenizer.get_vocab().items()}
    teacher_vocab = {k.replace("\u0120", "\u2581"): v for k, v in teacher_tokenizer.get_vocab().items()}

    teacher_vocab_size = len(teacher_vocab)
    student_vocab_size = len(student_vocab)
    t2s = torch.full((teacher_vocab_size,), -1, dtype=torch.long)
    s2t = torch.full((student_vocab_size,), -1, dtype=torch.long)

    overlap_tokens = []
    student_overlap_ids: list[int] = []
    teacher_overlap_ids: list[int] = []

    for token, tea_id in teacher_vocab.items():
        if token in student_vocab:
            stu_id = student_vocab[token]
            t2s[tea_id] = stu_id
            overlap_tokens.append(token)
            student_overlap_ids.append(stu_id)
            teacher_overlap_ids.append(tea_id)
    for token, stu_id in student_vocab.items():
        if token in teacher_vocab:
            s2t[stu_id] = teacher_vocab[token]

    # Force-map specials (eos/bos/pad).
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        tea_sid = getattr(teacher_tokenizer, attr, None)
        stu_sid = getattr(student_tokenizer, attr, None)
        if tea_sid is not None and stu_sid is not None:
            t2s[int(tea_sid)] = int(stu_sid)
            s2t[int(stu_sid)] = int(tea_sid)

    return t2s, s2t, student_overlap_ids, teacher_overlap_ids


def _maybe_grad_clamp_hook(value: float):
    """Return a hook function clamping gradient to ``[-value, value]``.

    Currently unused (projector is frozen). Kept for forward compatibility
    with a future trainable-projector revision.
    """
    def _hook(grad: torch.Tensor) -> torch.Tensor:
        return grad.clamp(-value, value)
    return _hook


def get_or_create_dskd_state(
    distillation_config,
    student_tokenizer,
    teacher_tokenizer,
    student_lm_head_weight: torch.Tensor,   # [V_stu, H_stu]
    teacher_lm_head: nn.Module,             # nn.Linear-like
    device: torch.device,
    dtype: torch.dtype,
) -> DSKDProjectorState:
    """Idempotently build the DSKD projector state.

    The projector is initialized via the closed-form pseudo-inverse solution
    so that ``teacher_logits @ pinv(student_lm_head)`` recovers a t2s mapping
    that, when re-projected through the student lm_head, approximates the
    original teacher logits.

    Args:
        distillation_config: full DistillationConfig (provides
            ``distillation_loss.dskd_topk_vocab``).
        student_tokenizer / teacher_tokenizer: hf tokenizers.
        student_lm_head_weight: ``[V_stu, H_stu]`` cloned, detached.
        teacher_lm_head: teacher lm_head module (so we can read ``.weight``).
        device / dtype: target device / dtype for the cached tensors.
    """
    global _STATE
    with _LOCK:
        if _STATE is not None and _STATE.device == device and _STATE.dtype == dtype:
            return _STATE

        loss_config = distillation_config.distillation_loss
        topk_vocab = int(getattr(loss_config, "dskd_topk_vocab", -1))
        token_align = str(getattr(loss_config, "dskd_token_align", "identical"))

        # Detach all ref tensors and move to device.
        student_lm_head = student_lm_head_weight.detach().to(device=device, dtype=dtype).contiguous()
        teacher_lm_head_w = teacher_lm_head.weight.detach().to(device=device, dtype=dtype).contiguous()

        # Vocab compatibility / identical detection.
        v_stu = student_lm_head.shape[0]
        v_tea = teacher_lm_head_w.shape[0]
        vocab_identical = (v_stu == v_tea) and (
            student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
        )

        # Build heads in the [hidden, vocab] layout used by KDFlow.
        student_head_hv = student_lm_head.transpose(0, 1).contiguous()    # [H_stu, V_stu]
        teacher_head_hv = teacher_lm_head_w.transpose(0, 1).contiguous()  # [H_tea, V_tea]

        if vocab_identical:
            if topk_vocab != -1:
                part_student_head = student_head_hv[:, :topk_vocab]
                part_teacher_head = teacher_head_hv[:, :topk_vocab]
            else:
                part_student_head = student_head_hv
                part_teacher_head = teacher_head_hv
            t2s_id_mapping = None
            s2t_id_mapping = None
            student_overlap_token_ids = None
            teacher_overlap_token_ids = None
        else:
            t2s_id_mapping, s2t_id_mapping, stu_overlap, tea_overlap = _build_id_mappings(
                student_tokenizer, teacher_tokenizer
            )
            student_overlap_token_ids = torch.tensor(stu_overlap, dtype=torch.long, device=device)
            teacher_overlap_token_ids = torch.tensor(tea_overlap, dtype=torch.long, device=device)
            part_student_head = student_head_hv.index_select(1, student_overlap_token_ids)
            part_teacher_head = teacher_head_hv.index_select(1, teacher_overlap_token_ids)
            if topk_vocab != -1:
                part_student_head = part_student_head[:, :topk_vocab]
                part_teacher_head = part_teacher_head[:, :topk_vocab]

        # Initialize t2s projector via pinv (in fp32 for numerical stability).
        logger.info("[EasyOPD:dskd] initializing t2s_projector via pseudo-inverse")
        part_student_head_pinv = torch.linalg.pinv(part_student_head.float())  # [V_part, H_stu]
        init_t2s = (part_teacher_head.float() @ part_student_head_pinv).transpose(0, 1)
        # init_t2s: [H_stu, H_tea]; but KDFlow stores it as nn.Linear(H_tea -> H_stu)
        # whose .weight is [H_stu, H_tea]. Same here.
        t2s_projector_weight = init_t2s.to(dtype=dtype, device=device).contiguous()

        # Pre-compute teacher_overlap_head pinv for s2t path.
        logger.info("[EasyOPD:dskd] pre-computing teacher_overlap_head pinv")
        teacher_overlap_head_pinv = torch.linalg.pinv(part_teacher_head.float())
        teacher_overlap_head_pinv = teacher_overlap_head_pinv.to(dtype=dtype, device=device).contiguous()

        # CMA-specific query projector (also frozen pinv-init).
        query_projector_weight = None
        if token_align == "cma":
            # KDFlow shape: nn.Linear(2*H_stu -> 2*H_tea), weight = [2*H_tea, 2*H_stu].
            # Initialize to a block-diagonal pinv so that the attention scores
            # behave like dot products in the joint space.
            h_stu = student_head_hv.shape[0]
            h_tea = teacher_head_hv.shape[0]
            block = torch.zeros((2 * h_tea, 2 * h_stu), dtype=torch.float32, device=device)
            # Block-diagonal: input top-half (stu_input_emb) -> output top-half (tea_input_emb-equivalent),
            # input bottom-half (stu_target_emb) -> output bottom-half. Use t2s_projector.T as a sensible init.
            t2s_T = t2s_projector_weight.float().transpose(0, 1)  # [H_tea, H_stu]
            block[:h_tea, :h_stu] = t2s_T
            block[h_tea:, h_stu:] = t2s_T
            query_projector_weight = block.to(dtype=dtype, device=device).contiguous()

        _STATE = DSKDProjectorState(
            student_lm_head=student_lm_head,
            teacher_lm_head=teacher_lm_head_w,
            t2s_projector_weight=t2s_projector_weight,
            teacher_overlap_head_pinv=teacher_overlap_head_pinv,
            t2s_id_mapping=t2s_id_mapping.to(device=device) if t2s_id_mapping is not None else None,
            s2t_id_mapping=s2t_id_mapping.to(device=device) if s2t_id_mapping is not None else None,
            student_overlap_token_ids=student_overlap_token_ids,
            teacher_overlap_token_ids=teacher_overlap_token_ids,
            query_projector_weight=query_projector_weight,
            vocab_identical=vocab_identical,
            device=device,
            dtype=dtype,
        )
        logger.info(
            "[EasyOPD:dskd] state ready: vocab_identical=%s, V_stu=%d V_tea=%d, "
            "token_align=%s, topk_vocab=%d",
            vocab_identical, v_stu, v_tea, token_align, topk_vocab,
        )
        return _STATE
