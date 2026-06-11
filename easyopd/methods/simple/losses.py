# Copyright 2026 EasyOPD Contributors
#
# Cross-tokenizer KD loss (`simple`) plugged into verl's on-policy
# distillation framework via the **logit-processor** protocol — same
# mechanism `forward_kl_topk` uses.
#
# Two-stage data flow:
#
#   Stage 1 (logit-processor, called inside the FSDP forward pass):
#     `compute_simple_xtok_logits_processor`
#       in : student_logits = [1, total_nnz, V_stu]   ← rmpad
#            data            : TensorDict (with non_tensor_batch attached)
#            cu_seqlens      : [B+1] int — sample boundaries within total_nnz
#       out: {"distillation_losses": [1, total_nnz]}  — per-token KL,
#            zero at prompt positions and at unaligned response positions.
#
#   Stage 2 (final policy loss, called inside `distillation_loss`):
#     `compute_distillation_loss_simple_cross_tokenizer`
#       reads `model_output["distillation_losses"]` (NestedTensor stored
#       by stage 1 via `torch.nested.nested_tensor_from_jagged`),
#       converts it to padded `[B, resp_len]`, returns metrics. Mirrors
#       `compute_forward_kl_topk` exactly.
#
# Per-sample teacher payload (numpy object arrays of length B, attached
# by the EasyOPD sidecar in `agent_loop._postprocess`):
#   - teacher_hidden_states  np.ndarray[N_i, H]   (last-layer, response only)
#   - teacher_input_ids      np.ndarray[T_i]      (full teacher sequence)
#   - teacher_loss_mask      np.ndarray[T_i] bool (response positions)

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import align_sequences, find_overlap_tokens

logger = logging.getLogger(__name__)


__all__ = [
    "compute_simple_xtok_logits_processor",
    "compute_distillation_loss_simple_cross_tokenizer",
    "register_simple_loss",
]


# ---------------------------------------------------------------------------
# Process-level singletons: teacher lm_head + tokenizers + overlap ids.
# Loaded lazily on the first call, on the device of the incoming logits.
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_TEACHER_LM_HEAD: Optional[nn.Linear] = None
_TEACHER_LM_HEAD_DEVICE: Optional[torch.device] = None
_TEACHER_TOKENIZER = None
_STUDENT_TOKENIZER = None
_STUDENT_OVERLAP_IDS: Optional[torch.Tensor] = None
_TEACHER_OVERLAP_IDS: Optional[torch.Tensor] = None


def _ensure_singletons(
    distillation_config,
    student_tokenizer_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Idempotently load teacher lm_head, tokenizers and overlap-id maps."""
    global _TEACHER_LM_HEAD, _TEACHER_LM_HEAD_DEVICE
    global _TEACHER_TOKENIZER, _STUDENT_TOKENIZER
    global _STUDENT_OVERLAP_IDS, _TEACHER_OVERLAP_IDS

    with _LOCK:
        if _TEACHER_LM_HEAD is None:
            from transformers import AutoTokenizer

            from .teacher_lm_head import load_teacher_lm_head

            teacher_models = distillation_config.teacher_models
            if len(teacher_models) != 1:
                raise NotImplementedError(
                    "[EasyOPD:simple] cross-tokenizer KD currently supports "
                    f"a single teacher; got {len(teacher_models)}."
                )
            _, teacher_cfg = next(iter(teacher_models.items()))
            teacher_path = teacher_cfg.model_path

            logger.warning(
                "[EasyOPD:simple] loading teacher lm_head from %s onto %s",
                teacher_path,
                device,
            )
            head = load_teacher_lm_head(teacher_path, dtype=dtype)
            head = head.to(device)
            head.requires_grad_(False)
            _TEACHER_LM_HEAD = head
            _TEACHER_LM_HEAD_DEVICE = device

            _TEACHER_TOKENIZER = AutoTokenizer.from_pretrained(
                teacher_path, trust_remote_code=True
            )
            if _TEACHER_TOKENIZER.pad_token is None:
                _TEACHER_TOKENIZER.pad_token = _TEACHER_TOKENIZER.eos_token
            _STUDENT_TOKENIZER = AutoTokenizer.from_pretrained(
                student_tokenizer_path, trust_remote_code=True
            )
            if _STUDENT_TOKENIZER.pad_token is None:
                _STUDENT_TOKENIZER.pad_token = _STUDENT_TOKENIZER.eos_token

            stu_ids, tea_ids = find_overlap_tokens(_STUDENT_TOKENIZER, _TEACHER_TOKENIZER)
            _STUDENT_OVERLAP_IDS = torch.tensor(stu_ids, dtype=torch.long, device=device)
            _TEACHER_OVERLAP_IDS = torch.tensor(tea_ids, dtype=torch.long, device=device)
            logger.warning(
                "[EasyOPD:simple] overlap vocab size: %d", _STUDENT_OVERLAP_IDS.numel()
            )
        elif _TEACHER_LM_HEAD_DEVICE != device:
            _TEACHER_LM_HEAD = _TEACHER_LM_HEAD.to(device)
            _TEACHER_LM_HEAD_DEVICE = device
            _STUDENT_OVERLAP_IDS = _STUDENT_OVERLAP_IDS.to(device)
            _TEACHER_OVERLAP_IDS = _TEACHER_OVERLAP_IDS.to(device)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_simple_loss() -> None:
    """Idempotently register the `simple` cross-tokenizer KD loss."""
    from verl.trainer.distillation.losses import (
        DISTILLATION_LOSS_REGISTRY,
        DistillationLossSettings,
        register_distillation_loss,
    )

    if "simple" in DISTILLATION_LOSS_REGISTRY:
        return

    decorator = register_distillation_loss(
        DistillationLossSettings(names=["simple"], use_cross_tokenizer=True)
    )
    decorator(compute_distillation_loss_simple_cross_tokenizer)


# ---------------------------------------------------------------------------
# KL helpers
# ---------------------------------------------------------------------------

def _kl_div(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """KL divergence on the overlap sub-vocabulary."""
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    if direction == "forward":
        student_p = student_logp.exp()
        kl = (student_p * (student_logp - teacher_logp)).sum(dim=-1)
    elif direction == "reverse":
        teacher_p = teacher_logp.exp()
        kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
    else:
        raise ValueError(
            f"Unsupported cross_tokenizer_kl_direction: {direction!r}."
        )
    return kl


def _resolve_student_path(config) -> str:
    """Locate the student tokenizer path. Prefers the env var set by
    `engine_workers.py` at student-worker init; falls back to attribute
    lookups (useful in unit tests)."""
    student_path = os.environ.get("EASYOPD_STUDENT_MODEL_PATH")
    if student_path is not None:
        return student_path
    student_path = getattr(config, "student_model_path", None) or getattr(
        config, "model_path", None
    )
    if student_path is None and hasattr(config, "model"):
        student_path = getattr(config.model, "path", None)
    if student_path is None:
        raise RuntimeError(
            "[EasyOPD:simple] cannot resolve student model path; "
            "set EASYOPD_STUDENT_MODEL_PATH environment variable."
        )
    return student_path


def _unwrap_non_tensor(arr):
    """Best-effort unwrap of verl `NonTensorData` / `NonTensorStack`
    wrappers into plain Python objects (e.g. a numpy object array or a
    list of numpy arrays). Pass-through for anything else."""
    if arr is None:
        return None
    # NonTensorStack: indexable per batch dim → list of unwrapped items
    if hasattr(arr, "tolist") and not isinstance(arr, (np.ndarray, torch.Tensor, list, tuple)):
        try:
            return arr.tolist()
        except Exception:  # noqa: BLE001
            pass
    # NonTensorData: exposes `.data`
    if hasattr(arr, "data") and not isinstance(arr, (np.ndarray, torch.Tensor)):
        return arr.data
    return arr


def _extract_non_tensor(data, key: str):
    """Pull a non-tensor entry out of a verl micro-batch.

    The micro-batch reaching the loss fn is a `TensorDict`; agent_loop
    populated `DataProto.non_tensor_batch[key]` (an `np.ndarray[object]`),
    which `DataProto.to_tensordict` re-wrapped into a `NonTensorStack`
    keyed by `key`. We unwrap that stack into a per-sample list of
    numpy arrays. As a fallback we also accept a `DataProto` (test
    contexts only)."""
    # Test path: caller passed a DataProto-like object.
    nt = getattr(data, "non_tensor_batch", None)
    if isinstance(nt, dict) and key in nt:
        return _unwrap_non_tensor(nt[key])

    # Production path: TensorDict → NonTensorStack.
    try:
        val = data.get(key)
    except Exception:  # noqa: BLE001
        val = None
    return _unwrap_non_tensor(val)


def _extract_response_lens(data, bsz: int, device) -> torch.Tensor:
    """Return `[B]` int tensor of per-sample response lengths.

    `data["response_mask"]` is a `(bsz, response_len)` NestedTensor whose
    offsets give cumulative response lengths. Falls back to summing along
    dim=1 for the dense layout."""
    rm = data["response_mask"]
    if rm.is_nested:
        offs = rm.offsets()  # [B+1]
        lens = offs.diff().to(device=device, dtype=torch.long)
    else:
        lens = rm.bool().sum(dim=1).to(device=device, dtype=torch.long)
    if lens.numel() != bsz:
        raise RuntimeError(
            f"[EasyOPD:simple] response_mask batch size {lens.numel()} "
            f"does not match cu_seqlens batch size {bsz}."
        )
    return lens


def _extract_input_ids_rmpad(data, total_nnz: int) -> torch.Tensor:
    """Get a `[total_nnz]` long tensor of token ids in rmpad layout."""
    ids = data["input_ids"]
    if ids.is_nested:
        flat = ids.values()
    else:
        flat = ids.reshape(-1)
    if flat.numel() != total_nnz:
        raise RuntimeError(
            f"[EasyOPD:simple] input_ids rmpad length {flat.numel()} "
            f"does not match student_logits total_nnz {total_nnz}."
        )
    return flat


# ---------------------------------------------------------------------------
# Stage 1: logit-processor (does the heavy lifting on student logits)
# ---------------------------------------------------------------------------

def compute_simple_xtok_logits_processor(
    student_logits: torch.Tensor,
    data,
    cu_seqlens: torch.Tensor,
    config,
    distillation_config,
) -> torch.Tensor:
    """Compute per-token cross-tokenizer KL **inside the forward pass**,
    while student_logits is still resident.

    Args:
        student_logits: `[1, total_nnz, V_stu]` rmpad logits, post-temperature.
        data:           TensorDict / DataProto with non_tensor_batch attached.
        cu_seqlens:     `[B+1]` int — rmpad sample boundaries.
        config:         ActorConfig.
        distillation_config: DistillationConfig.

    Returns:
        `[1, total_nnz]` float tensor — per-token KL on the overlap
        sub-vocab; zero at prompt positions and at unaligned response
        positions. Caller (fsdp `prepare_model_outputs`) wraps this into a
        NestedTensor and stuffs it into `model_output["distillation_losses"]`.
    """
    assert student_logits.dim() == 3 and student_logits.shape[0] == 1, (
        f"expected student_logits [1, total_nnz, V], got {tuple(student_logits.shape)}"
    )
    total_nnz = student_logits.shape[1]
    device = student_logits.device
    dtype = student_logits.dtype

    student_path = _resolve_student_path(config)
    _ensure_singletons(
        distillation_config=distillation_config,
        student_tokenizer_path=student_path,
        device=device,
        dtype=dtype,
    )
    teacher_lm_head = _TEACHER_LM_HEAD
    teacher_tokenizer = _TEACHER_TOKENIZER
    student_tokenizer = _STUDENT_TOKENIZER
    student_overlap_ids = _STUDENT_OVERLAP_IDS
    teacher_overlap_ids = _TEACHER_OVERLAP_IDS

    teacher_hidden_states_arr = _extract_non_tensor(data, "teacher_hidden_states")
    teacher_input_ids_arr = _extract_non_tensor(data, "teacher_input_ids")
    teacher_loss_mask_arr = _extract_non_tensor(data, "teacher_loss_mask")
    if teacher_hidden_states_arr is None:
        raise KeyError(
            "[EasyOPD:simple] teacher_hidden_states not found; "
            "did the agent_loop sidecar populate non_tensor_batch?"
        )

    response_lens = _extract_response_lens(data, bsz=len(cu_seqlens) - 1, device=device)
    input_ids_rmpad = _extract_input_ids_rmpad(data, total_nnz)

    loss_config = distillation_config.distillation_loss
    direction = getattr(loss_config, "cross_tokenizer_kl_direction", "forward")

    # Output buffer in rmpad layout (zeros = no signal).
    out = torch.zeros((total_nnz,), dtype=dtype, device=device)

    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    bsz = len(cu) - 1
    resp_lens_cpu = response_lens.detach().cpu().tolist()

    total_aligned = 0
    total_response = 0

    for b in range(bsz):
        s_lo, s_hi = int(cu[b]), int(cu[b + 1])
        sample_len = s_hi - s_lo
        if sample_len == 0:
            continue
        stu_resp_len = int(resp_lens_cpu[b])
        if stu_resp_len == 0:
            continue
        if stu_resp_len > sample_len:
            raise RuntimeError(
                f"[EasyOPD:simple] sample {b}: response_len {stu_resp_len} > "
                f"sample_len {sample_len} (input_ids rmpad slice)."
            )
        # NO_PADDING layout: response segment occupies the trailing
        # `resp_len` tokens within the sample's rmpad slice.
        prompt_len = sample_len - stu_resp_len
        resp_lo_global = s_lo + prompt_len
        resp_hi_global = s_hi  # exclusive
        total_response += stu_resp_len

        # Teacher hidden states are selected at label/logit positions:
        # position i predicts token i+1. For alignment, compare those label
        # tokens; for loss placement, write KL back to the predicting logits.
        tea_label_hs_np = np.asarray(teacher_hidden_states_arr[b])
        if tea_label_hs_np is None or tea_label_hs_np.shape[0] == 0:
            continue
        tea_label_hs_np = np.asarray(tea_label_hs_np, dtype=np.float32)
        tea_input_ids_np = np.asarray(teacher_input_ids_arr[b], dtype=np.int64)
        tea_loss_mask_np = np.asarray(teacher_loss_mask_arr[b], dtype=bool)
        tea_logit_pos = np.nonzero(tea_loss_mask_np)[0]
        tea_label_pos = tea_logit_pos + 1
        valid_tea = tea_label_pos < len(tea_input_ids_np)
        tea_logit_pos = tea_logit_pos[valid_tea]
        tea_label_pos = tea_label_pos[valid_tea]
        tea_label_hs_np = tea_label_hs_np[valid_tea]
        if len(tea_label_pos) == 0:
            continue
        tea_label_ids = tea_input_ids_np[tea_label_pos].tolist()

        # Student label tokens are the response tokens, while the logits that
        # predict them occupy [resp_lo_global - 1, resp_hi_global - 1).
        stu_label_ids = input_ids_rmpad[resp_lo_global:resp_hi_global].detach().cpu().tolist()
        stu_logit_global = torch.arange(
            resp_lo_global - 1,
            resp_hi_global - 1,
            dtype=torch.long,
            device=device,
        )
        valid_stu = stu_logit_global >= s_lo
        if not bool(valid_stu.any().item()):
            continue
        stu_logit_global = stu_logit_global[valid_stu]
        stu_label_ids = [tok for tok, keep in zip(stu_label_ids, valid_stu.detach().cpu().tolist()) if keep]

        # Greedy character-level alignment over label tokens.
        stu_tokens = student_tokenizer.convert_ids_to_tokens(stu_label_ids)
        tea_tokens = teacher_tokenizer.convert_ids_to_tokens(tea_label_ids)
        tea_align_idx, stu_align_idx = align_sequences(
            tea_tokens,
            stu_tokens,
            teacher_eos_token=getattr(teacher_tokenizer, "eos_token", None),
            student_eos_token=getattr(student_tokenizer, "eos_token", None),
        )
        if not stu_align_idx:
            continue
        total_aligned += len(stu_align_idx)

        stu_local = torch.tensor(stu_align_idx, dtype=torch.long, device=device)
        tea_local = torch.tensor(tea_align_idx, dtype=torch.long, device=device)

        # Absolute rmpad logit positions for the aligned student label tokens.
        stu_abs_global = stu_logit_global.index_select(0, stu_local)

        # Project teacher hidden → overlap logits.
        tea_hs = torch.from_numpy(tea_label_hs_np).to(device=device, dtype=dtype)
        tea_hs_aligned = tea_hs.index_select(0, tea_local)              # [N, H]
        with torch.no_grad():
            tea_full_logits = teacher_lm_head(tea_hs_aligned)           # [N, V_tea]
        tea_logits_overlap = tea_full_logits.index_select(
            -1, teacher_overlap_ids
        )                                                                # [N, K]

        # Student: gather the aligned logit positions, then column-crop.
        stu_logits_full = student_logits[0].index_select(0, stu_abs_global)
        stu_logits_overlap = stu_logits_full.index_select(
            -1, student_overlap_ids
        )

        if stu_logits_overlap.shape != tea_logits_overlap.shape:
            raise RuntimeError(
                "Aligned student/teacher overlap-logits shape mismatch: "
                f"student={tuple(stu_logits_overlap.shape)} vs "
                f"teacher={tuple(tea_logits_overlap.shape)}"
            )

        kl_per_pos = _kl_div(stu_logits_overlap, tea_logits_overlap, direction)
        out.index_copy_(0, stu_abs_global, kl_per_pos.to(dtype))

    align_ratio = float(total_aligned) / max(total_response, 1)
    if align_ratio < 0.5:
        logger.warning(
            "[EasyOPD:simple] align_ratio=%.3f below 0.5; KD signal may be weak.",
            align_ratio,
        )

    return out.unsqueeze(0)  # [1, total_nnz]


# ---------------------------------------------------------------------------
# Stage 2: final policy loss assembly (mirrors compute_forward_kl_topk)
# ---------------------------------------------------------------------------

def compute_distillation_loss_simple_cross_tokenizer(
    config,
    distillation_config,
    model_output: dict,
    data,
) -> Tuple[torch.Tensor, dict]:
    """Final policy-loss-side wrapper.

    Stage 1 has already populated `model_output["distillation_losses"]`
    with a NestedTensor of per-token KL values (zeros outside response /
    unaligned positions). We just convert to padded `[B, resp_len]` and
    return metrics. Heavy lifting lives in
    `compute_simple_xtok_logits_processor`.
    """
    from verl.workers.utils.padding import no_padding_2_padding

    if "distillation_losses" not in model_output:
        raise KeyError(
            "[EasyOPD:simple] model_output['distillation_losses'] missing — "
            "stage-1 logit processor was not invoked. Check that "
            "DistillationLossSettings(use_cross_tokenizer=True) was registered "
            "and that the FSDP engine routes through "
            "`compute_simple_xtok_logits_processor`."
        )

    distillation_losses = no_padding_2_padding(
        model_output["distillation_losses"], data
    )

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == response_mask_bool.shape, (
        f"shape mismatch: distillation_losses={tuple(distillation_losses.shape)} "
        f"vs response_mask={tuple(response_mask_bool.shape)}"
    )

    # Forward KL is non-negative; reverse KL too. Numerical noise can push
    # values slightly below zero in fp16/bf16 — clamp to be safe.
    distillation_losses = distillation_losses.clamp_min(0.0)

    from verl.utils.metric import AggregationType, Metric

    metrics: dict = {
        "distillation/overlap_vocab_size": Metric(
            AggregationType.MEAN,
            torch.tensor(
                float(_STUDENT_OVERLAP_IDS.numel()) if _STUDENT_OVERLAP_IDS is not None else 0.0
            ),
        ),
    }
    return distillation_losses, metrics