# Copyright 2026 EasyOPD Contributors
#
# SimCT (`span_ctkd`) loss registration and verl logit-processor entry points.
#
# This module ports KDFlow's `span_ctkd` algorithm into verl's cross-tokenizer
# distillation path. It intentionally reuses the EasyOPD `simple` teacher
# sidecar, teacher lm_head and overlap-vocabulary singleton setup; only the
# student/teacher alignment and loss construction differ.
#
# SimCT-v2 changes (vs original SimCT v1):
#   1. Each aligned segment uses overlap-vocabulary KL plus a span dimension
#      for multi-token segments. The span dimension aggregates each token's
#      ground-truth logit using **mean**.
#   2. First-token masking is **retained** to avoid double-counting the first
#      token in both the overlap and span dimensions.
#   3. The per-segment KL clamp (`simct_loss_clamp`) is removed; the global
#      `simple_loss_clamp` in `verl/dp_actor.py` (10.0) still acts as a guard.
#   4. Segment alignment (`align_label_ids_with_spans`) is unchanged and
#      keeps the relaxed cumulative-text matching.

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SIMCT_LOSS_NAMES = ("simct", "span_ctkd")
EOS_MARKER = "<|EASYOPD_SIMCT_EOS|>"
DEBUG_SIMCT = os.getenv("EASYOPD_SIMCT_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_simct_loss() -> None:
    """Idempotently register `simct` and legacy `span_ctkd` loss names."""
    from verl.trainer.distillation.losses import (
        DISTILLATION_LOSS_REGISTRY,
        DistillationLossSettings,
        register_distillation_loss,
    )

    if all(name in DISTILLATION_LOSS_REGISTRY for name in SIMCT_LOSS_NAMES):
        return

    names_to_register = [name for name in SIMCT_LOSS_NAMES if name not in DISTILLATION_LOSS_REGISTRY]
    if not names_to_register:
        return

    decorator = register_distillation_loss(
        DistillationLossSettings(names=names_to_register, use_cross_tokenizer=True)
    )
    decorator(compute_distillation_loss_simct_cross_tokenizer)


# ---------------------------------------------------------------------------
# Token/span alignment
# ---------------------------------------------------------------------------

def _to_int_list(ids: Iterable[int] | torch.Tensor | np.ndarray) -> list[int]:
    if torch.is_tensor(ids):
        return [int(x) for x in ids.detach().cpu().tolist()]
    if isinstance(ids, np.ndarray):
        return [int(x) for x in ids.tolist()]
    return [int(x) for x in ids]


def decode_token_texts(token_ids: Sequence[int], tokenizer) -> list[str]:
    """Decode each token id independently for cross-tokenizer alignment.

    This deliberately uses ``tokenizer.decode([tid])`` instead of
    ``convert_ids_to_tokens`` so byte-level BPE newline markers, spaces and
    tokenizer-specific normalization are preserved.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    texts: list[str] = []
    for tid in token_ids:
        tid = int(tid)
        if eos_id is not None and tid == int(eos_id):
            texts.append(EOS_MARKER)
        elif tid in special_ids:
            texts.append("")
        else:
            texts.append(tokenizer.decode([tid], skip_special_tokens=False))
    return texts


def align_label_ids_with_spans(
    teacher_label_ids: Sequence[int],
    student_label_ids: Sequence[int],
    teacher_tokenizer,
    student_tokenizer,
) -> tuple[list[tuple[int, int, int, int]], list[int], list[int]]:
    """Align teacher/student label ids and identify span segments.

    Args:
        teacher_label_ids: Teacher next-token labels selected by teacher
            loss-mask positions.
        student_label_ids: Student next-token labels selected by student
            loss-mask positions.
        teacher_tokenizer: Teacher tokenizer.
        student_tokenizer: Student tokenizer.

    Returns:
        ``(segments, teacher_ids, student_ids)``. Each segment is
        ``(tea_start, tea_end, stu_start, stu_end)`` with local ``[start, end)``
        indices into the label-id lists. A segment is a span when either side
        covers more than one token.
    """
    teacher_ids = _to_int_list(teacher_label_ids)
    student_ids = _to_int_list(student_label_ids)
    if not teacher_ids or not student_ids:
        return [], teacher_ids, student_ids

    teacher_texts = decode_token_texts(teacher_ids, teacher_tokenizer)
    student_texts = decode_token_texts(student_ids, student_tokenizer)

    teacher_idx = 0
    student_idx = 0
    teacher_start = 0
    student_start = 0
    teacher_history = ""
    student_history = ""
    segments: list[tuple[int, int, int, int]] = []

    while teacher_idx < len(teacher_texts) or student_idx < len(student_texts):
        if teacher_history == student_history and teacher_history:
            segments.append((teacher_start, teacher_idx, student_start, student_idx))
            teacher_start = teacher_idx
            student_start = student_idx
            teacher_history = ""
            student_history = ""
            continue

        if teacher_idx >= len(teacher_texts):
            if student_idx >= len(student_texts):
                break
            student_history += student_texts[student_idx]
            student_idx += 1
            continue
        if student_idx >= len(student_texts):
            teacher_history += teacher_texts[teacher_idx]
            teacher_idx += 1
            continue

        if not teacher_history and not student_history:
            teacher_history += teacher_texts[teacher_idx]
            student_history += student_texts[student_idx]
            teacher_idx += 1
            student_idx += 1
        elif len(teacher_history) <= len(student_history):
            teacher_history += teacher_texts[teacher_idx]
            teacher_idx += 1
        else:
            student_history += student_texts[student_idx]
            student_idx += 1

    if teacher_history == student_history and teacher_history:
        segments.append((teacher_start, teacher_idx, student_start, student_idx))

    return segments, teacher_ids, student_ids


# ---------------------------------------------------------------------------
# Virtual common vocabulary logits
# ---------------------------------------------------------------------------

# Module-level lookup caches for overlap id -> position mapping.
# Built once per unique overlap_ids tensor (keyed by data_ptr).
_STUDENT_OVERLAP_ID2POS: dict[int, int] | None = None
_TEACHER_OVERLAP_ID2POS: dict[int, int] | None = None
_STUDENT_OVERLAP_PTR: int = 0
_TEACHER_OVERLAP_PTR: int = 0


def _get_overlap_id2pos(overlap_ids: torch.Tensor, cache_key: str) -> dict[int, int]:
    """Get or build a {token_id -> position_in_overlap} lookup dict.

    This replaces O(vocab_size) `(overlap_ids == token_id).nonzero()` calls
    with O(1) dict lookups after a one-time O(overlap_size) build.
    """
    global _STUDENT_OVERLAP_ID2POS, _TEACHER_OVERLAP_ID2POS
    global _STUDENT_OVERLAP_PTR, _TEACHER_OVERLAP_PTR

    ptr = overlap_ids.data_ptr()
    if cache_key == "student":
        if _STUDENT_OVERLAP_ID2POS is not None and _STUDENT_OVERLAP_PTR == ptr:
            return _STUDENT_OVERLAP_ID2POS
        id2pos = {}
        for pos, tid in enumerate(overlap_ids.tolist()):
            if tid not in id2pos:  # keep first occurrence
                id2pos[tid] = pos
        _STUDENT_OVERLAP_ID2POS = id2pos
        _STUDENT_OVERLAP_PTR = ptr
        return id2pos
    else:
        if _TEACHER_OVERLAP_ID2POS is not None and _TEACHER_OVERLAP_PTR == ptr:
            return _TEACHER_OVERLAP_ID2POS
        id2pos = {}
        for pos, tid in enumerate(overlap_ids.tolist()):
            if tid not in id2pos:
                id2pos[tid] = pos
        _TEACHER_OVERLAP_ID2POS = id2pos
        _TEACHER_OVERLAP_PTR = ptr
        return id2pos


def build_virtual_vocab_logits(
    segments: Sequence[tuple[int, int, int, int]],
    student_logits_aligned: torch.Tensor,
    teacher_logits_aligned: torch.Tensor,
    student_label_ids: Sequence[int],
    teacher_label_ids: Sequence[int],
    student_overlap_ids: torch.Tensor,
    teacher_overlap_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
    """Build virtual-vocabulary logits for aligned segments.

    For each segment the first logit position is projected to the shared
    overlap vocabulary. For **span segments** (where either side covers more
    than one token), an additional span dimension is appended whose value is
    the **mean** of each token's logit for its own ground-truth id within the
    span. The first token's overlap position is masked to ``-1e9`` to avoid
    double-counting it in both the overlap and the span dimension.
    """
    num_overlap = int(student_overlap_ids.numel())
    device = student_logits_aligned.device
    dtype = student_logits_aligned.dtype

    # Build O(1) lookup tables for first-token masking (cached across calls)
    stu_id2pos = _get_overlap_id2pos(student_overlap_ids, "student")
    tea_id2pos = _get_overlap_id2pos(teacher_overlap_ids, "teacher")

    span_segment_indices = [
        idx
        for idx, (tea_start, tea_end, stu_start, stu_end) in enumerate(segments)
        if (tea_end - tea_start) > 1 or (stu_end - stu_start) > 1
    ]
    segment_to_span_dim = {seg_idx: dim for dim, seg_idx in enumerate(span_segment_indices)}
    num_spans = len(span_segment_indices)

    student_rows: list[torch.Tensor] = []
    teacher_rows: list[torch.Tensor] = []
    is_span_segment: list[bool] = []

    for seg_idx, (tea_start, tea_end, stu_start, stu_end) in enumerate(segments):
        student_segment_logits = student_logits_aligned[stu_start:stu_end]
        teacher_segment_logits = teacher_logits_aligned[tea_start:tea_end]
        if student_segment_logits.numel() == 0 or teacher_segment_logits.numel() == 0:
            continue

        student_overlap = student_segment_logits[0].index_select(0, student_overlap_ids).clone()
        teacher_overlap = teacher_segment_logits[0].index_select(0, teacher_overlap_ids).clone()
        current_is_span = seg_idx in segment_to_span_dim

        # First-token masking: mask the first token's position in overlap to
        # avoid counting it twice (once in overlap, once in span dim).
        if current_is_span:
            first_stu_token = int(student_label_ids[stu_start])
            first_tea_token = int(teacher_label_ids[tea_start])
            stu_pos = stu_id2pos.get(first_stu_token)
            if stu_pos is not None:
                student_overlap[stu_pos] = -1e9
            tea_pos = tea_id2pos.get(first_tea_token)
            if tea_pos is not None:
                teacher_overlap[tea_pos] = -1e9

        if num_spans > 0:
            student_span_dims = torch.full((num_spans,), -1e9, device=device, dtype=dtype)
            teacher_span_dims = torch.full((num_spans,), -1e9, device=device, dtype=teacher_overlap.dtype)

            if current_is_span:
                span_dim = segment_to_span_dim[seg_idx]
                # Span logit = MEAN of all tokens' logits for their own
                # ground-truth ids within the span.
                student_self_logits = []
                for local_idx, token_id in enumerate(student_label_ids[stu_start:stu_end]):
                    student_self_logits.append(student_segment_logits[local_idx, int(token_id)])
                teacher_self_logits = []
                for local_idx, token_id in enumerate(teacher_label_ids[tea_start:tea_end]):
                    teacher_self_logits.append(teacher_segment_logits[local_idx, int(token_id)])
                if student_self_logits:
                    student_span_dims[span_dim] = torch.stack(student_self_logits).mean()
                if teacher_self_logits:
                    teacher_span_dims[span_dim] = torch.stack(teacher_self_logits).mean()

            student_row = torch.cat([student_overlap, student_span_dims], dim=0)
            teacher_row = torch.cat([teacher_overlap, teacher_span_dims], dim=0)
        else:
            student_row = student_overlap
            teacher_row = teacher_overlap

        if student_row.numel() != num_overlap + num_spans:
            raise RuntimeError(
                f"[EasyOPD:simct] invalid virtual row dim {student_row.numel()} "
                f"for expected dim {num_overlap + num_spans}."
            )
        student_rows.append(student_row)
        teacher_rows.append(teacher_row)
        is_span_segment.append(current_is_span)

    if not student_rows:
        empty = torch.empty((0, num_overlap + num_spans), device=device, dtype=dtype)
        return empty, empty, []

    return torch.stack(student_rows, dim=0), torch.stack(teacher_rows, dim=0), is_span_segment


# ---------------------------------------------------------------------------
# Processor helpers
# ---------------------------------------------------------------------------

def _teacher_label_ids_from_mask(
    teacher_input_ids: np.ndarray,
    teacher_loss_mask: np.ndarray,
) -> list[int]:
    mask = np.asarray(teacher_loss_mask).astype(bool)
    label_positions = np.nonzero(mask)[0] + 1
    label_positions = label_positions[label_positions < len(teacher_input_ids)]
    return [int(teacher_input_ids[pos]) for pos in label_positions]


def _response_mask_bool(data) -> torch.Tensor:
    if data["response_mask"].is_nested:
        return data["response_mask"].bool().to_padded_tensor(False)
    return data["response_mask"].bool()


# ---------------------------------------------------------------------------
# Stage 1: logit processor
# ---------------------------------------------------------------------------

def compute_simct_xtok_logits_processor(
    student_logits: torch.Tensor,
    data,
    cu_seqlens: torch.Tensor,
    config,
    distillation_config,
) -> dict[str, torch.Tensor]:
    """Compute SimCT span-level loss inside the student forward pass."""
    assert student_logits.dim() == 3 and student_logits.shape[0] == 1, (
        f"expected student_logits [1, total_nnz, V], got {tuple(student_logits.shape)}"
    )

    from easyopd.methods.simple import losses as simple_losses

    total_nnz = student_logits.shape[1]
    device = student_logits.device
    dtype = student_logits.dtype

    student_path = simple_losses._resolve_student_path(config)
    simple_losses._ensure_singletons(
        distillation_config=distillation_config,
        student_tokenizer_path=student_path,
        device=device,
        dtype=dtype,
    )

    teacher_lm_head = simple_losses._TEACHER_LM_HEAD
    teacher_tokenizer = simple_losses._TEACHER_TOKENIZER
    student_tokenizer = simple_losses._STUDENT_TOKENIZER
    student_overlap_ids = simple_losses._STUDENT_OVERLAP_IDS
    teacher_overlap_ids = simple_losses._TEACHER_OVERLAP_IDS
    if (
        teacher_lm_head is None
        or teacher_tokenizer is None
        or student_tokenizer is None
        or student_overlap_ids is None
        or teacher_overlap_ids is None
    ):
        raise RuntimeError("[EasyOPD:simct] shared simple singletons were not initialized.")

    teacher_hidden_states_arr = simple_losses._extract_non_tensor(data, "teacher_hidden_states")
    teacher_input_ids_arr = simple_losses._extract_non_tensor(data, "teacher_input_ids")
    teacher_loss_mask_arr = simple_losses._extract_non_tensor(data, "teacher_loss_mask")
    if teacher_hidden_states_arr is None:
        raise KeyError(
            "[EasyOPD:simct] teacher_hidden_states not found; "
            "did the agent_loop sidecar populate non_tensor_batch?"
        )
    if teacher_input_ids_arr is None or teacher_loss_mask_arr is None:
        raise KeyError(
            "[EasyOPD:simct] teacher_input_ids/teacher_loss_mask missing; "
            "cannot construct teacher label ids for span alignment."
        )

    input_ids_rmpad = simple_losses._extract_input_ids_rmpad(data, total_nnz)
    response_lens = simple_losses._extract_response_lens(data, bsz=len(cu_seqlens) - 1, device=device)

    loss_config = distillation_config.distillation_loss
    direction = getattr(loss_config, "cross_tokenizer_kl_direction", "reverse")

    out = torch.zeros((total_nnz,), dtype=dtype, device=device)
    valid_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)
    span_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)
    response_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)

    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    response_lens_cpu = response_lens.detach().cpu().tolist()

    total_segments = 0
    total_span_segments = 0
    total_response_tokens = 0
    skipped_samples = 0
    # Task 5: Collect per-segment KL values for debug logging
    _debug_span_kls: list[float] = []
    _debug_overlap_kls: list[float] = []

    for batch_idx in range(len(cu) - 1):
        sample_start, sample_end = int(cu[batch_idx]), int(cu[batch_idx + 1])
        sample_len = sample_end - sample_start
        if sample_len <= 1:
            skipped_samples += 1
            continue

        response_len = int(response_lens_cpu[batch_idx])
        if response_len <= 0:
            skipped_samples += 1
            continue
        if response_len >= sample_len:
            # Need one prompt-side logit position to predict the first response token.
            logger.warning(
                "[EasyOPD:simct] sample %d skipped because response_len=%d >= sample_len=%d.",
                batch_idx,
                response_len,
                sample_len,
            )
            skipped_samples += 1
            continue

        # Verl's response losses are stored at logits positions that predict
        # response labels: [last_prompt_token, ..., penultimate_response_token].
        student_loss_start = sample_end - response_len - 1
        student_loss_end = sample_end - 1
        student_abs_positions = torch.arange(
            student_loss_start, student_loss_end, dtype=torch.long, device=device
        )
        if student_abs_positions.numel() == 0:
            skipped_samples += 1
            continue
        response_positions.index_fill_(0, student_abs_positions, 1.0)
        total_response_tokens += int(student_abs_positions.numel())

        student_label_ids = (
            input_ids_rmpad.index_select(0, student_abs_positions + 1)
            .detach()
            .cpu()
            .tolist()
        )
        student_logits_aligned = student_logits[0].index_select(0, student_abs_positions)

        teacher_hidden_np: np.ndarray = teacher_hidden_states_arr[batch_idx]
        teacher_ids_np: np.ndarray = teacher_input_ids_arr[batch_idx]
        teacher_mask_np: np.ndarray = teacher_loss_mask_arr[batch_idx]
        if teacher_hidden_np is None or teacher_hidden_np.shape[0] == 0:
            skipped_samples += 1
            continue

        teacher_label_ids = _teacher_label_ids_from_mask(teacher_ids_np, teacher_mask_np)
        if not teacher_label_ids:
            skipped_samples += 1
            continue
        if len(teacher_label_ids) != int(teacher_hidden_np.shape[0]):
            usable = min(len(teacher_label_ids), int(teacher_hidden_np.shape[0]))
            logger.warning(
                "[EasyOPD:simct] sample %d teacher label/hidden length mismatch: "
                "labels=%d hidden=%d; truncating to %d.",
                batch_idx,
                len(teacher_label_ids),
                int(teacher_hidden_np.shape[0]),
                usable,
            )
            teacher_label_ids = teacher_label_ids[:usable]
            teacher_hidden_np = teacher_hidden_np[:usable]

        segments, teacher_ids_list, student_ids_list = align_label_ids_with_spans(
            teacher_label_ids=teacher_label_ids,
            student_label_ids=student_label_ids,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
        )
        if not segments:
            if DEBUG_SIMCT:
                logger.warning(
                    "[EasyOPD:simct debug] sample=%d skipped: no aligned segments "
                    "student_labels=%d teacher_labels=%d response_len=%d.",
                    batch_idx,
                    len(student_label_ids),
                    len(teacher_label_ids),
                    response_len,
                )
            skipped_samples += 1
            continue

        teacher_hidden = torch.from_numpy(np.ascontiguousarray(teacher_hidden_np)).to(device=device, dtype=dtype)
        with torch.no_grad():
            teacher_logits_aligned = teacher_lm_head(teacher_hidden)

        student_virtual, teacher_virtual, is_span_segment = build_virtual_vocab_logits(
            segments=segments,
            student_logits_aligned=student_logits_aligned,
            teacher_logits_aligned=teacher_logits_aligned,
            student_label_ids=student_ids_list,
            teacher_label_ids=teacher_ids_list,
            student_overlap_ids=student_overlap_ids,
            teacher_overlap_ids=teacher_overlap_ids,
        )
        if student_virtual.numel() == 0:
            skipped_samples += 1
            continue
        if student_virtual.shape != teacher_virtual.shape:
            raise RuntimeError(
                "[EasyOPD:simct] virtual logits shape mismatch: "
                f"student={tuple(student_virtual.shape)} vs teacher={tuple(teacher_virtual.shape)}"
            )
        if DEBUG_SIMCT:
            logger.info(
                "[EasyOPD:simct debug] sample=%d segments=%d span_segments=%d "
                "student_labels=%d teacher_labels=%d virtual_dim=%d.",
                batch_idx,
                len(segments),
                sum(1 for flag in is_span_segment if flag),
                len(student_ids_list),
                len(teacher_ids_list),
                student_virtual.shape[-1],
            )

        kl_per_segment = simple_losses._kl_div(student_virtual, teacher_virtual.detach(), direction)
        # SimCT-v2: per-segment clamp removed; global simple_loss_clamp in dp_actor handles guarding.
        if kl_per_segment.numel() != len(is_span_segment):
            raise RuntimeError(
                "[EasyOPD:simct] segment loss count mismatch: "
                f"losses={kl_per_segment.numel()} spans={len(is_span_segment)}."
            )

        for seg_idx, (_, _, student_seg_start, _) in enumerate(segments[: kl_per_segment.numel()]):
            if student_seg_start >= student_abs_positions.numel():
                continue
            target_pos = student_abs_positions[int(student_seg_start)]
            out[target_pos] = kl_per_segment[seg_idx].to(dtype)
            valid_positions[target_pos] = 1.0
            if is_span_segment[seg_idx]:
                span_positions[target_pos] = 1.0

        # Task 5: Collect KL values per segment type for debug stats
        if DEBUG_SIMCT:
            for seg_idx_dbg, is_span_flag in enumerate(is_span_segment):
                kl_val = float(kl_per_segment[seg_idx_dbg].item())
                if is_span_flag:
                    _debug_span_kls.append(kl_val)
                else:
                    _debug_overlap_kls.append(kl_val)

        total_segments += int(kl_per_segment.numel())
        total_span_segments += int(sum(1 for flag in is_span_segment if flag))

    # Task 5: Debug log — compare span vs 1:1 segment KL magnitudes
    if DEBUG_SIMCT and (_debug_span_kls or _debug_overlap_kls):
        avg_span_kl = sum(_debug_span_kls) / max(len(_debug_span_kls), 1)
        avg_overlap_kl = sum(_debug_overlap_kls) / max(len(_debug_overlap_kls), 1)
        ratio = avg_span_kl / max(avg_overlap_kl, 1e-8)
        logger.info(
            "[EasyOPD:simct debug] KL stats: span_avg=%.4f (%d segs), "
            "overlap_avg=%.4f (%d segs), ratio=%.2fx",
            avg_span_kl, len(_debug_span_kls),
            avg_overlap_kl, len(_debug_overlap_kls),
            ratio,
        )

    if total_response_tokens > 0 and total_segments == 0:
        logger.warning(
            "[EasyOPD:simct] no valid aligned segments in batch; skipped_samples=%d.",
            skipped_samples,
        )
    elif total_response_tokens > 0:
        logger.debug(
            "[EasyOPD:simct] segments=%d span_segments=%d response_tokens=%d skipped_samples=%d.",
            total_segments,
            total_span_segments,
            total_response_tokens,
            skipped_samples,
        )

    stats_dtype = dtype
    total_segments_tensor = torch.full((total_nnz,), float(total_segments), dtype=stats_dtype, device=device)
    total_span_segments_tensor = torch.full((total_nnz,), float(total_span_segments), dtype=stats_dtype, device=device)
    skipped_samples_tensor = torch.full((total_nnz,), float(skipped_samples), dtype=stats_dtype, device=device)

    return {
        "distillation_losses": out.unsqueeze(0),
        "simct_valid_positions": valid_positions.unsqueeze(0),
        "simct_span_positions": span_positions.unsqueeze(0),
        "simct_response_positions": response_positions.unsqueeze(0),
        "simct_total_segments": total_segments_tensor.unsqueeze(0),
        "simct_total_span_segments": total_span_segments_tensor.unsqueeze(0),
        "simct_skipped_samples": skipped_samples_tensor.unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# Stage 2: final policy loss assembly
# ---------------------------------------------------------------------------

def compute_distillation_loss_simct_cross_tokenizer(
    config,
    distillation_config,
    model_output: dict,
    data,
) -> Tuple[torch.Tensor, dict[str, Any]]:
    """Convert SimCT rmpad losses back to response layout and report metrics.

    SimCT-v2 metric notes:
        * ``distillation/overlap_vocab_size`` reports the overlap vocabulary
          size (== virtual vocab dim in v2; the span dimension is removed).
        * ``distillation/simct_span_segments`` is preserved for backward
          compatibility but now only counts span-shaped segments for
          diagnostics; their KL contribution comes from pure overlap KL,
          not a dedicated span dimension.
    """
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    from easyopd.methods.simple import losses as simple_losses

    if "distillation_losses" not in model_output:
        raise KeyError(
            "[EasyOPD:simct] model_output['distillation_losses'] missing — "
            "stage-1 SimCT logit processor was not invoked."
        )

    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data).clamp_min(0.0)
    response_mask_bool = _response_mask_bool(data)
    if distillation_losses.shape != response_mask_bool.shape:
        raise RuntimeError(
            "[EasyOPD:simct] shape mismatch: "
            f"distillation_losses={tuple(distillation_losses.shape)} vs "
            f"response_mask={tuple(response_mask_bool.shape)}"
        )

    if "simct_valid_positions" in model_output:
        valid_positions = no_padding_2_padding(model_output["simct_valid_positions"], data) > 0
    else:
        valid_positions = distillation_losses > 0
    if "simct_span_positions" in model_output:
        span_positions = no_padding_2_padding(model_output["simct_span_positions"], data) > 0
    else:
        span_positions = torch.zeros_like(response_mask_bool)
    if "simct_response_positions" in model_output:
        simct_response_positions = no_padding_2_padding(model_output["simct_response_positions"], data) > 0
    else:
        simct_response_positions = response_mask_bool

    response_tokens = (simct_response_positions & response_mask_bool).float().sum().clamp_min(1.0)
    valid_count = (valid_positions & response_mask_bool).float().sum()
    span_count = (span_positions & response_mask_bool).float().sum()
    all_invalid = valid_count <= 0
    active_losses = distillation_losses[response_mask_bool]
    loss_mean = active_losses.mean() if active_losses.numel() > 0 else distillation_losses.new_tensor(0.0)
    if active_losses.numel() > 0:
        simct_loss_sum = active_losses.sum()
    else:
        simct_loss_sum = distillation_losses.new_tensor(0.0)

    overlap_size = 0.0
    if simple_losses._STUDENT_OVERLAP_IDS is not None:
        overlap_size = float(simple_losses._STUDENT_OVERLAP_IDS.numel())

    total_segments_value = valid_count
    if "simct_total_segments" in model_output:
        total_segments_value = no_padding_2_padding(model_output["simct_total_segments"], data)[0, 0]
    total_span_segments_value = span_count
    if "simct_total_span_segments" in model_output:
        total_span_segments_value = no_padding_2_padding(model_output["simct_total_span_segments"], data)[0, 0]
    skipped_samples_value = distillation_losses.new_tensor(0.0)
    if "simct_skipped_samples" in model_output:
        skipped_samples_value = no_padding_2_padding(model_output["simct_skipped_samples"], data)[0, 0]

    metrics: dict[str, Any] = {
        "distillation/overlap_vocab_size": Metric(
            AggregationType.MEAN,
            distillation_losses.new_tensor(overlap_size),
        ),
        "distillation/simct_valid_segments": Metric(AggregationType.SUM, valid_count),
        "distillation/simct_span_segments": Metric(AggregationType.SUM, span_count),
        "distillation/simct_total_segments": Metric(AggregationType.SUM, total_segments_value),
        "distillation/simct_total_span_segments": Metric(AggregationType.SUM, total_span_segments_value),
        "distillation/simct_skipped_samples": Metric(AggregationType.SUM, skipped_samples_value),
        "distillation/simct_all_invalid_batch": Metric(
            AggregationType.MEAN,
            distillation_losses.new_tensor(float(bool(all_invalid))),
        ),
        "distillation/simct_valid_span_ratio": Metric(
            AggregationType.MEAN,
            valid_count / response_tokens,
        ),
        "distillation/simct_loss": Metric(AggregationType.SUM, simct_loss_sum),
        "distillation/simct_loss_mean": Metric(AggregationType.MEAN, loss_mean),
    }
    return distillation_losses, metrics


__all__ = [
    "SIMCT_LOSS_NAMES",
    "align_label_ids_with_spans",
    "build_virtual_vocab_logits",
    "compute_distillation_loss_simct_cross_tokenizer",
    "compute_simct_xtok_logits_processor",
    "decode_token_texts",
    "register_simct_loss",
]
