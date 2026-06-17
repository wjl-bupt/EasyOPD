# Copyright 2026 EasyOPD Contributors
#
# ALM (Approximate Likelihood Matching) cross-tokenizer KD loss, ported from
# KDFlow's `kdflow.algorithms.alm` to verl's logit-processor protocol.
#
# Two-stage data flow (mirrors `easyopd.methods.simct.losses`):
#
#   Stage 1 — `compute_alm_xtok_logits_processor`:
#     in : student_logits = [1, total_nnz, V_stu]  (rmpad)
#          data            : TensorDict with non_tensor_batch attached
#                            (teacher_hidden_states / teacher_input_ids /
#                            teacher_loss_mask)
#          cu_seqlens      : [B+1] int — sample boundaries within total_nnz
#     out: dict {
#            "distillation_losses":      [1, total_nnz],
#            "alm_valid_positions":      [1, total_nnz],
#            "alm_response_positions":   [1, total_nnz],
#            "alm_total_chunks":         [1, total_nnz] (broadcast scalar),
#            "alm_skipped_samples":      [1, total_nnz] (broadcast scalar),
#          }
#     Per-chunk binarised f-divergence loss is placed on the first student
#     logit position of the chunk; all other positions are 0.
#
#   Stage 2 — `compute_distillation_loss_alm_cross_tokenizer`:
#     converts back to padded `[B, response_len]`, clamps to >= 0, returns
#     metrics under `distillation/alm_*`.

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ALM_LOSS_NAMES = ("alm",)
EOS_MARKER = "<|EASYOPD_ALM_EOS|>"


__all__ = [
    "ALM_LOSS_NAMES",
    "compute_alm_xtok_logits_processor",
    "compute_distillation_loss_alm_cross_tokenizer",
    "register_alm_loss",
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_alm_loss() -> None:
    """Idempotently register the `alm` cross-tokenizer KD loss."""
    from verl.trainer.distillation.losses import (
        DISTILLATION_LOSS_REGISTRY,
        DistillationLossSettings,
        register_distillation_loss,
    )

    if all(name in DISTILLATION_LOSS_REGISTRY for name in ALM_LOSS_NAMES):
        return

    names_to_register = [n for n in ALM_LOSS_NAMES if n not in DISTILLATION_LOSS_REGISTRY]
    if not names_to_register:
        return

    decorator = register_distillation_loss(
        DistillationLossSettings(names=names_to_register, use_cross_tokenizer=True)
    )
    decorator(compute_distillation_loss_alm_cross_tokenizer)


# ---------------------------------------------------------------------------
# Chunk alignment (equivalent to KDFlow's `alm._compute_chunk_alignment`)
# ---------------------------------------------------------------------------

def _decode_per_token(token_ids, tokenizer) -> list[str]:
    """Decode each token id independently to obtain its text contribution.

    Mirrors KDFlow `alm.decode_tokens`: EOS -> sentinel, other special tokens
    -> empty string, ordinary tokens -> ``tokenizer.decode([tid])``.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    out: list[str] = []
    for tid in token_ids:
        tid = int(tid)
        if eos_id is not None and tid == int(eos_id):
            out.append(EOS_MARKER)
        elif tid in special_ids:
            out.append("")
        else:
            out.append(tokenizer.decode([tid], skip_special_tokens=False))
    return out


def _compute_chunk_alignment_local(
    tea_label_ids: list[int],
    stu_label_ids: list[int],
    teacher_tokenizer,
    student_tokenizer,
) -> list[tuple[int, int, int, int]]:
    """Greedy cumulative-text chunk alignment over already-extracted
    *response label* token id lists.

    Returns chunks ``(tea_start, tea_end, stu_start, stu_end)`` in *local*
    indices (into the input lists). Each chunk's range is ``[start, end)``
    with ``end > start``. The semantics match KDFlow exactly: a "boundary"
    is a position where teacher cumulative text == student cumulative text
    AND the next teacher/student token text is identical.
    """
    if not tea_label_ids or not stu_label_ids:
        return []

    tea_texts = _decode_per_token(tea_label_ids, teacher_tokenizer)
    stu_texts = _decode_per_token(stu_label_ids, student_tokenizer)

    boundaries: list[tuple[int, int]] = []
    i, j = 0, 0
    history_tea = ""
    history_stu = ""

    while i < len(tea_texts) and j < len(stu_texts):
        if history_tea == history_stu and tea_texts[i] == stu_texts[j]:
            boundaries.append((i, j))
            history_tea += tea_texts[i]
            history_stu += stu_texts[j]
            i += 1
            j += 1
        elif len(history_tea) > len(history_stu):
            history_stu += stu_texts[j]
            j += 1
        elif len(history_tea) < len(history_stu):
            history_tea += tea_texts[i]
            i += 1
        else:
            history_tea += tea_texts[i]
            history_stu += stu_texts[j]
            i += 1
            j += 1

    if not boundaries:
        return []

    chunks: list[tuple[int, int, int, int]] = []
    for idx in range(len(boundaries)):
        if idx == 0:
            local_tea_start, local_stu_start = 0, 0
        else:
            local_tea_start = boundaries[idx - 1][0] + 1
            local_stu_start = boundaries[idx - 1][1] + 1
        local_tea_end = boundaries[idx][0] + 1
        local_stu_end = boundaries[idx][1] + 1
        if local_tea_end > local_tea_start and local_stu_end > local_stu_start:
            chunks.append((local_tea_start, local_tea_end, local_stu_start, local_stu_end))

    return chunks


# ---------------------------------------------------------------------------
# Binarised f-divergence (equivalent to KDFlow `alm._binarised_f_divergence`)
# ---------------------------------------------------------------------------

def _binarised_f_divergence(
    log_p_teacher: torch.Tensor,
    log_p_student: torch.Tensor,
    temperature: float,
    f_divergence: str,
) -> torch.Tensor:
    """Compute per-chunk binarised f-divergence.

    For ``temperature >= 50``, uses the closed-form ``tau -> infinity``
    approximation matching KDFlow exactly. Otherwise computes the binarised
    KL / TVD via the ``p^{1/tau}`` parameterization.
    """
    tau = float(temperature)

    if f_divergence == "tvd":
        if tau >= 50.0:
            return 2.0 * torch.abs(log_p_teacher - log_p_student)
        p_t = torch.exp(log_p_teacher / tau)
        p_s = torch.exp(log_p_student / tau)
        return torch.abs(p_t - p_s) + torch.abs((1 - p_t) - (1 - p_s))

    if f_divergence == "kl":
        if tau >= 50.0:
            term1 = log_p_teacher - log_p_student
            eps = 1e-8
            log_p_t_safe = log_p_teacher.clamp(max=-eps)
            log_p_s_safe = log_p_student.clamp(max=-eps)
            term2 = log_p_t_safe * torch.log(log_p_s_safe / log_p_t_safe)
            return term1 + term2
        p_t = torch.exp(log_p_teacher / tau).clamp(1e-8, 1 - 1e-8)
        p_s = torch.exp(log_p_student / tau).clamp(1e-8, 1 - 1e-8)
        kl_pos = p_t * torch.log(p_t / p_s)
        kl_neg = (1 - p_t) * torch.log((1 - p_t) / (1 - p_s))
        return kl_pos + kl_neg

    raise ValueError(f"Unknown f-divergence: {f_divergence!r}; expected 'kl' or 'tvd'.")


# ---------------------------------------------------------------------------
# Stage 1 — logit processor
# ---------------------------------------------------------------------------

def compute_alm_xtok_logits_processor(
    student_logits: torch.Tensor,
    data,
    cu_seqlens: torch.Tensor,
    config,
    distillation_config,
) -> dict[str, torch.Tensor]:
    """Compute ALM chunk-level loss inside the student forward pass.

    Returns a dict whose ``"distillation_losses"`` key is a ``[1, total_nnz]``
    rmpad tensor with per-chunk loss placed at the *first* student logit
    position of each chunk.
    """
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
    if teacher_lm_head is None or teacher_tokenizer is None or student_tokenizer is None:
        raise RuntimeError("[EasyOPD:alm] shared simple singletons not initialized.")

    teacher_hidden_states_arr = simple_losses._extract_non_tensor(data, "teacher_hidden_states")
    teacher_input_ids_arr = simple_losses._extract_non_tensor(data, "teacher_input_ids")
    teacher_loss_mask_arr = simple_losses._extract_non_tensor(data, "teacher_loss_mask")
    if teacher_hidden_states_arr is None:
        raise KeyError(
            "[EasyOPD:alm] teacher_hidden_states not found in non_tensor_batch."
        )
    if teacher_input_ids_arr is None or teacher_loss_mask_arr is None:
        raise KeyError(
            "[EasyOPD:alm] teacher_input_ids/teacher_loss_mask missing; cannot align."
        )

    input_ids_rmpad = simple_losses._extract_input_ids_rmpad(data, total_nnz)
    response_lens = simple_losses._extract_response_lens(data, bsz=len(cu_seqlens) - 1, device=device)

    loss_config = distillation_config.distillation_loss
    alm_temperature = float(getattr(loss_config, "alm_temperature", 100.0))
    alm_f_divergence = str(getattr(loss_config, "alm_f_divergence", "kl"))

    out = torch.zeros((total_nnz,), dtype=dtype, device=device)
    valid_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)
    response_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)

    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    response_lens_cpu = response_lens.detach().cpu().tolist()

    total_chunks = 0
    skipped_samples = 0

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
            logger.warning(
                "[EasyOPD:alm] sample %d skipped (response_len=%d >= sample_len=%d).",
                batch_idx, response_len, sample_len,
            )
            skipped_samples += 1
            continue

        # Student logit positions that predict response labels.
        student_loss_start = sample_end - response_len - 1
        student_loss_end = sample_end - 1
        student_abs_positions = torch.arange(
            student_loss_start, student_loss_end, dtype=torch.long, device=device,
        )
        if student_abs_positions.numel() == 0:
            skipped_samples += 1
            continue
        response_positions.index_fill_(0, student_abs_positions, 1.0)

        # Student response label ids: input_ids at logit_pos + 1.
        stu_label_positions = student_abs_positions + 1
        stu_label_ids = (
            input_ids_rmpad.index_select(0, stu_label_positions).detach().cpu().tolist()
        )
        student_logits_aligned = student_logits[0].index_select(0, student_abs_positions)

        teacher_hidden_np: np.ndarray = teacher_hidden_states_arr[batch_idx]
        teacher_ids_np: np.ndarray = np.asarray(teacher_input_ids_arr[batch_idx], dtype=np.int64)
        teacher_mask_np: np.ndarray = np.asarray(teacher_loss_mask_arr[batch_idx], dtype=bool)
        if teacher_hidden_np is None or int(teacher_hidden_np.shape[0]) == 0:
            skipped_samples += 1
            continue
        teacher_hidden_np = np.asarray(teacher_hidden_np, dtype=np.float32)

        # Teacher: for every loss-mask position k, label is at k+1 (KDFlow
        # _compute_chunk_log_probs_loss_region semantics).
        tea_logit_positions = np.nonzero(teacher_mask_np)[0]
        tea_label_positions = tea_logit_positions + 1
        valid_tea = tea_label_positions < len(teacher_ids_np)
        tea_logit_positions = tea_logit_positions[valid_tea]
        tea_label_positions = tea_label_positions[valid_tea]
        if len(tea_label_positions) == 0:
            skipped_samples += 1
            continue
        # Truncate to min(len(label), hidden_rows) — guards against
        # length mismatch like simct does.
        usable = min(len(tea_label_positions), int(teacher_hidden_np.shape[0]))
        if usable != len(tea_label_positions) or usable != int(teacher_hidden_np.shape[0]):
            logger.warning(
                "[EasyOPD:alm] sample %d teacher label/hidden length mismatch: "
                "labels=%d hidden=%d; truncating to %d.",
                batch_idx, len(tea_label_positions), int(teacher_hidden_np.shape[0]), usable,
            )
        tea_label_positions = tea_label_positions[:usable]
        teacher_hidden_np = teacher_hidden_np[:usable]
        tea_label_ids = teacher_ids_np[tea_label_positions].tolist()

        # Compute chunks in *local* indices (into the label-id lists).
        chunks_local = _compute_chunk_alignment_local(
            tea_label_ids=tea_label_ids,
            stu_label_ids=stu_label_ids,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
        )
        if not chunks_local:
            skipped_samples += 1
            continue

        # Teacher logits for the loss-mask region, [usable, V_tea].
        teacher_hidden = torch.from_numpy(teacher_hidden_np).to(device=device, dtype=dtype)
        with torch.no_grad():
            teacher_logits_loss = teacher_lm_head(teacher_hidden)
        # Teacher token log-probs at each loss-mask position.
        tea_log_probs_full = F.log_softmax(teacher_logits_loss.float(), dim=-1)
        tea_label_ids_t = torch.tensor(tea_label_ids, dtype=torch.long, device=device)
        tea_token_log_probs = tea_log_probs_full.gather(
            -1, tea_label_ids_t.unsqueeze(-1)
        ).squeeze(-1)  # [usable]

        # Student token log-probs at exactly the chunk-needed positions.
        # Collect all needed local student positions first.
        all_stu_local_positions: list[int] = []
        for _, _, stu_s, stu_e in chunks_local:
            all_stu_local_positions.extend(range(stu_s, stu_e))
        if not all_stu_local_positions:
            skipped_samples += 1
            continue
        stu_pos_t = torch.tensor(all_stu_local_positions, dtype=torch.long, device=device)
        # student_logits_aligned is [response_len, V_stu]; index_select gives
        # [num_needed, V_stu], then we log_softmax only over needed rows.
        needed_student_logits = student_logits_aligned.index_select(0, stu_pos_t)
        stu_log_probs = F.log_softmax(needed_student_logits.float(), dim=-1)
        # Labels for those positions: stu_label_ids list, indexed locally.
        needed_stu_labels = torch.tensor(
            [stu_label_ids[p] for p in all_stu_local_positions],
            dtype=torch.long, device=device,
        )
        stu_token_log_probs = stu_log_probs.gather(
            -1, needed_stu_labels.unsqueeze(-1)
        ).squeeze(-1)  # [num_needed]

        # Sum into per-chunk log-probs.
        tea_chunk_lps: list[torch.Tensor] = []
        stu_chunk_lps: list[torch.Tensor] = []
        chunk_first_stu_local: list[int] = []
        offset = 0
        for tea_s, tea_e, stu_s, stu_e in chunks_local:
            stu_len = stu_e - stu_s
            stu_lp = stu_token_log_probs[offset:offset + stu_len].sum()
            offset += stu_len
            tea_lp = tea_token_log_probs[tea_s:tea_e].sum()
            tea_chunk_lps.append(tea_lp)
            stu_chunk_lps.append(stu_lp)
            chunk_first_stu_local.append(stu_s)

        if not tea_chunk_lps:
            skipped_samples += 1
            continue

        tea_lp_t = torch.stack(tea_chunk_lps).detach()  # detach teacher
        stu_lp_t = torch.stack(stu_chunk_lps)
        chunk_losses = _binarised_f_divergence(
            tea_lp_t, stu_lp_t, alm_temperature, alm_f_divergence
        )  # [num_chunks]

        # Place each chunk loss at the chunk's first student logit position.
        for ci, stu_local in enumerate(chunk_first_stu_local):
            if stu_local >= student_abs_positions.numel():
                continue
            target_pos = student_abs_positions[int(stu_local)]
            out[target_pos] = chunk_losses[ci].to(dtype)
            valid_positions[target_pos] = 1.0

        total_chunks += len(chunks_local)

    stats_dtype = dtype
    total_chunks_t = torch.full((total_nnz,), float(total_chunks), dtype=stats_dtype, device=device)
    skipped_samples_t = torch.full((total_nnz,), float(skipped_samples), dtype=stats_dtype, device=device)

    return {
        "distillation_losses": out.unsqueeze(0),
        "alm_valid_positions": valid_positions.unsqueeze(0),
        "alm_response_positions": response_positions.unsqueeze(0),
        "alm_total_chunks": total_chunks_t.unsqueeze(0),
        "alm_skipped_samples": skipped_samples_t.unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# Stage 2 — final policy-loss assembly
# ---------------------------------------------------------------------------

def compute_distillation_loss_alm_cross_tokenizer(
    config,
    distillation_config,
    model_output: dict,
    data,
) -> Tuple[torch.Tensor, dict[str, Any]]:
    """Convert ALM rmpad losses back to padded layout and report metrics."""
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    if "distillation_losses" not in model_output:
        raise KeyError(
            "[EasyOPD:alm] model_output['distillation_losses'] missing — "
            "stage-1 ALM logit processor was not invoked."
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
            "[EasyOPD:alm] shape mismatch: "
            f"distillation_losses={tuple(distillation_losses.shape)} vs "
            f"response_mask={tuple(response_mask_bool.shape)}"
        )

    if "alm_valid_positions" in model_output:
        valid_positions = no_padding_2_padding(model_output["alm_valid_positions"], data) > 0
    else:
        valid_positions = distillation_losses > 0
    if "alm_response_positions" in model_output:
        response_positions = no_padding_2_padding(model_output["alm_response_positions"], data) > 0
    else:
        response_positions = response_mask_bool

    response_tokens = (response_positions & response_mask_bool).float().sum().clamp_min(1.0)
    valid_count = (valid_positions & response_mask_bool).float().sum()
    active_losses = distillation_losses[response_mask_bool]
    loss_mean = active_losses.mean() if active_losses.numel() > 0 else distillation_losses.new_tensor(0.0)
    loss_sum = active_losses.sum() if active_losses.numel() > 0 else distillation_losses.new_tensor(0.0)

    total_chunks_value = valid_count
    if "alm_total_chunks" in model_output:
        total_chunks_value = no_padding_2_padding(model_output["alm_total_chunks"], data)[0, 0]
    skipped_samples_value = distillation_losses.new_tensor(0.0)
    if "alm_skipped_samples" in model_output:
        skipped_samples_value = no_padding_2_padding(model_output["alm_skipped_samples"], data)[0, 0]

    loss_config = distillation_config.distillation_loss
    alm_temperature = float(getattr(loss_config, "alm_temperature", 100.0))
    alm_f_divergence = str(getattr(loss_config, "alm_f_divergence", "kl"))
    f_div_code = 0.0 if alm_f_divergence == "kl" else 1.0

    metrics: dict[str, Any] = {
        "distillation/alm_loss": Metric(AggregationType.SUM, loss_sum),
        "distillation/alm_loss_mean": Metric(AggregationType.MEAN, loss_mean),
        "distillation/alm_chunks": Metric(AggregationType.SUM, total_chunks_value),
        "distillation/alm_align_ratio": Metric(
            AggregationType.MEAN, valid_count / response_tokens
        ),
        "distillation/alm_skipped_samples": Metric(AggregationType.SUM, skipped_samples_value),
        "distillation/alm_temperature": Metric(
            AggregationType.MEAN,
            distillation_losses.new_tensor(alm_temperature),
        ),
        "distillation/alm_f_divergence_is_tvd": Metric(
            AggregationType.MEAN,
            distillation_losses.new_tensor(f_div_code),
        ),
    }
    return distillation_losses, metrics
