# Copyright 2026 EasyOPD Contributors
#
# ULD (Universal Logit Distillation) cross-tokenizer KD loss, ported from
# KDFlow `kdflow.algorithms.uld` to verl's logit-processor protocol.
#
# Two-stage data flow (mirrors `easyopd.methods.simct.losses`):
#
#   Stage 1 — `compute_uld_xtok_logits_processor`:
#     For each sample, greedily align teacher vs student response tokens
#     (cumulative-text comparison after `▁`/`Ġ` normalization), extract the
#     aligned student/teacher logits, compute Wasserstein-1 distance
#     between sorted probability vectors (top-k approximation by default),
#     and place per-aligned-position W1 onto the student logit position.
#
#   Stage 2 — `compute_distillation_loss_uld_cross_tokenizer`:
#     Convert rmpad -> padded `[B, response_len]`, scale by `uld_lambda`,
#     report metrics under `distillation/uld_*`.

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from easyopd.methods._align_utils import align_token_sequences

logger = logging.getLogger(__name__)

ULD_LOSS_NAMES = ("uld",)


__all__ = [
    "ULD_LOSS_NAMES",
    "compute_uld_xtok_logits_processor",
    "compute_distillation_loss_uld_cross_tokenizer",
    "register_uld_loss",
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_uld_loss() -> None:
    """Idempotently register the `uld` cross-tokenizer KD loss."""
    from verl.trainer.distillation.losses import (
        DISTILLATION_LOSS_REGISTRY,
        DistillationLossSettings,
        register_distillation_loss,
    )

    if all(name in DISTILLATION_LOSS_REGISTRY for name in ULD_LOSS_NAMES):
        return

    names_to_register = [n for n in ULD_LOSS_NAMES if n not in DISTILLATION_LOSS_REGISTRY]
    if not names_to_register:
        return

    decorator = register_distillation_loss(
        DistillationLossSettings(names=names_to_register, use_cross_tokenizer=True)
    )
    decorator(compute_distillation_loss_uld_cross_tokenizer)


# ---------------------------------------------------------------------------
# Wasserstein-1 (closed-form, top-k approximation)
# ---------------------------------------------------------------------------

def _compute_wasserstein_1(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Closed-form W1 between sorted teacher/student probability vectors.

    Equivalent to KDFlow `uld._compute_wasserstein_1`. Uses top-k
    approximation by default (residual mass aggregated into a single bin)
    to avoid sort/pad over very large vocabularies (e.g. 256K).

    Args:
        student_logits: ``[N, V_s]`` logits.
        teacher_logits: ``[N, V_t]`` logits.
        temperature: softmax temperature (uniform over both).
        top_k: top-k bucket size; if ``<= 0`` or ``>= min(V_s, V_t)`` falls
            back to full-vocab sort+pad.

    Returns:
        ``[N]`` per-position W1 distances.
    """
    student_probs = F.softmax(student_logits / temperature, dim=-1, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)

    vocab_s = student_probs.shape[-1]
    vocab_t = teacher_probs.shape[-1]

    if top_k > 0 and top_k < min(vocab_s, vocab_t):
        stu_topk_vals, _ = student_probs.topk(top_k, dim=-1, sorted=True)
        tea_topk_vals, _ = teacher_probs.topk(top_k, dim=-1, sorted=True)
        stu_residual = (1.0 - stu_topk_vals.sum(dim=-1, keepdim=True)).clamp(min=0)
        tea_residual = (1.0 - tea_topk_vals.sum(dim=-1, keepdim=True)).clamp(min=0)
        student_sorted = torch.cat([stu_topk_vals, stu_residual], dim=-1)
        teacher_sorted = torch.cat([tea_topk_vals, tea_residual], dim=-1)
        del student_probs, teacher_probs, stu_topk_vals, tea_topk_vals
    else:
        max_vocab = max(vocab_s, vocab_t)
        if vocab_s < max_vocab:
            pad = torch.zeros(
                student_probs.shape[0], max_vocab - vocab_s,
                device=student_probs.device, dtype=student_probs.dtype,
            )
            student_probs = torch.cat([student_probs, pad], dim=-1)
        if vocab_t < max_vocab:
            pad = torch.zeros(
                teacher_probs.shape[0], max_vocab - vocab_t,
                device=teacher_probs.device, dtype=teacher_probs.dtype,
            )
            teacher_probs = torch.cat([teacher_probs, pad], dim=-1)
        student_sorted, _ = student_probs.sort(dim=-1, descending=True)
        teacher_sorted, _ = teacher_probs.sort(dim=-1, descending=True)

    return torch.abs(student_sorted - teacher_sorted).sum(dim=-1)


# ---------------------------------------------------------------------------
# Stage 1 — logit processor
# ---------------------------------------------------------------------------

def compute_uld_xtok_logits_processor(
    student_logits: torch.Tensor,
    data,
    cu_seqlens: torch.Tensor,
    config,
    distillation_config,
) -> dict[str, torch.Tensor]:
    """Compute ULD per-position W1 inside the student forward pass."""
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
        raise RuntimeError("[EasyOPD:uld] shared simple singletons not initialized.")

    teacher_hidden_states_arr = simple_losses._extract_non_tensor(data, "teacher_hidden_states")
    teacher_input_ids_arr = simple_losses._extract_non_tensor(data, "teacher_input_ids")
    teacher_loss_mask_arr = simple_losses._extract_non_tensor(data, "teacher_loss_mask")
    if teacher_hidden_states_arr is None:
        raise KeyError("[EasyOPD:uld] teacher_hidden_states missing in non_tensor_batch.")
    if teacher_input_ids_arr is None or teacher_loss_mask_arr is None:
        raise KeyError("[EasyOPD:uld] teacher_input_ids/teacher_loss_mask missing.")

    input_ids_rmpad = simple_losses._extract_input_ids_rmpad(data, total_nnz)
    response_lens = simple_losses._extract_response_lens(data, bsz=len(cu_seqlens) - 1, device=device)

    loss_config = distillation_config.distillation_loss
    uld_lambda = float(getattr(loss_config, "uld_lambda", 1.5))
    uld_temperature = float(getattr(loss_config, "uld_temperature", 1.0))
    uld_top_k = int(getattr(loss_config, "uld_top_k", 1024))

    out = torch.zeros((total_nnz,), dtype=dtype, device=device)
    valid_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)
    response_positions = torch.zeros((total_nnz,), dtype=dtype, device=device)

    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    response_lens_cpu = response_lens.detach().cpu().tolist()

    total_aligned = 0
    total_response = 0
    skipped_samples = 0

    tea_eos = getattr(teacher_tokenizer, "eos_token", None)
    stu_eos = getattr(student_tokenizer, "eos_token", None)

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
                "[EasyOPD:uld] sample %d skipped (response_len=%d >= sample_len=%d).",
                batch_idx, response_len, sample_len,
            )
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
        total_response += int(student_abs_positions.numel())

        # Student response label ids and aligned logits.
        stu_label_positions = student_abs_positions + 1
        stu_label_ids = (
            input_ids_rmpad.index_select(0, stu_label_positions).detach().cpu().tolist()
        )
        student_logits_aligned = student_logits[0].index_select(0, student_abs_positions)

        # Teacher side.
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

        # Token-level alignment over response label tokens.
        tea_tokens = teacher_tokenizer.convert_ids_to_tokens(tea_label_ids)
        stu_tokens = student_tokenizer.convert_ids_to_tokens(stu_label_ids)
        tea_align_idx, stu_align_idx = align_token_sequences(
            tea_tokens, stu_tokens,
            teacher_eos_token=tea_eos,
            student_eos_token=stu_eos,
        )
        if not tea_align_idx or not stu_align_idx:
            skipped_samples += 1
            continue

        # Align tensors.
        tea_align_t = torch.tensor(tea_align_idx, dtype=torch.long, device=device)
        stu_align_t = torch.tensor(stu_align_idx, dtype=torch.long, device=device)
        # Teacher: hidden -> logits at aligned positions only.
        teacher_hidden = torch.from_numpy(teacher_hidden_np).to(device=device, dtype=dtype)
        teacher_hidden_aligned = teacher_hidden.index_select(0, tea_align_t)
        with torch.no_grad():
            teacher_logits_aligned = teacher_lm_head(teacher_hidden_aligned)
        # Student: logits at aligned local positions.
        student_logits_at_align = student_logits_aligned.index_select(0, stu_align_t)

        w1_per_pos = _compute_wasserstein_1(
            student_logits_at_align,
            teacher_logits_aligned.detach(),
            temperature=uld_temperature,
            top_k=uld_top_k,
        )  # [num_aligned], float32

        scaled = (uld_lambda * w1_per_pos).to(dtype)
        # Place at the aligned student logit positions in the rmpad layout.
        target_pos = student_abs_positions.index_select(0, stu_align_t)
        out.index_copy_(0, target_pos, scaled)
        valid_positions.index_fill_(0, target_pos, 1.0)

        total_aligned += int(stu_align_t.numel())

    if total_response > 0 and total_aligned == 0:
        logger.warning(
            "[EasyOPD:uld] no aligned positions in batch; skipped_samples=%d.",
            skipped_samples,
        )

    stats_dtype = dtype
    skipped_samples_t = torch.full((total_nnz,), float(skipped_samples), dtype=stats_dtype, device=device)
    total_aligned_t = torch.full((total_nnz,), float(total_aligned), dtype=stats_dtype, device=device)
    total_response_t = torch.full((total_nnz,), float(total_response), dtype=stats_dtype, device=device)

    return {
        "distillation_losses": out.unsqueeze(0),
        "uld_valid_positions": valid_positions.unsqueeze(0),
        "uld_response_positions": response_positions.unsqueeze(0),
        "uld_total_aligned": total_aligned_t.unsqueeze(0),
        "uld_total_response": total_response_t.unsqueeze(0),
        "uld_skipped_samples": skipped_samples_t.unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# Stage 2 — final policy-loss assembly
# ---------------------------------------------------------------------------

def compute_distillation_loss_uld_cross_tokenizer(
    config,
    distillation_config,
    model_output: dict,
    data,
) -> Tuple[torch.Tensor, dict[str, Any]]:
    """Convert ULD rmpad losses back to padded layout and report metrics."""
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    if "distillation_losses" not in model_output:
        raise KeyError(
            "[EasyOPD:uld] model_output['distillation_losses'] missing — "
            "stage-1 ULD logit processor was not invoked."
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
            "[EasyOPD:uld] shape mismatch: "
            f"distillation_losses={tuple(distillation_losses.shape)} vs "
            f"response_mask={tuple(response_mask_bool.shape)}"
        )

    if "uld_valid_positions" in model_output:
        valid_positions = no_padding_2_padding(model_output["uld_valid_positions"], data) > 0
    else:
        valid_positions = distillation_losses > 0
    if "uld_response_positions" in model_output:
        response_positions = no_padding_2_padding(model_output["uld_response_positions"], data) > 0
    else:
        response_positions = response_mask_bool

    response_tokens = (response_positions & response_mask_bool).float().sum().clamp_min(1.0)
    valid_count = (valid_positions & response_mask_bool).float().sum()
    active_losses = distillation_losses[response_mask_bool]
    loss_mean = active_losses.mean() if active_losses.numel() > 0 else distillation_losses.new_tensor(0.0)
    loss_sum = active_losses.sum() if active_losses.numel() > 0 else distillation_losses.new_tensor(0.0)

    skipped_samples_value = distillation_losses.new_tensor(0.0)
    if "uld_skipped_samples" in model_output:
        skipped_samples_value = no_padding_2_padding(model_output["uld_skipped_samples"], data)[0, 0]

    loss_config = distillation_config.distillation_loss
    uld_lambda = float(getattr(loss_config, "uld_lambda", 1.5))
    uld_temperature = float(getattr(loss_config, "uld_temperature", 1.0))
    uld_top_k = int(getattr(loss_config, "uld_top_k", 1024))

    metrics: dict[str, Any] = {
        "distillation/uld_loss": Metric(AggregationType.SUM, loss_sum),
        "distillation/uld_loss_mean": Metric(AggregationType.MEAN, loss_mean),
        "distillation/uld_align_ratio": Metric(
            AggregationType.MEAN, valid_count / response_tokens
        ),
        "distillation/uld_skipped_samples": Metric(AggregationType.SUM, skipped_samples_value),
        "distillation/uld_lambda": Metric(
            AggregationType.MEAN, distillation_losses.new_tensor(uld_lambda)
        ),
        "distillation/uld_temperature": Metric(
            AggregationType.MEAN, distillation_losses.new_tensor(uld_temperature)
        ),
        "distillation/uld_top_k": Metric(
            AggregationType.MEAN, distillation_losses.new_tensor(float(uld_top_k))
        ),
    }
    return distillation_losses, metrics
