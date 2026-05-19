# Copyright 2026 EasyOPD Contributors
#
# Cross-tokenizer KD loss (`simple`) plugged into verl's on-policy
# distillation framework.
#
# Data flow (after teacher sidecar refactor):
#
#     1. AgentLoopWorker._postprocess collects per-sample teacher hidden
#        states from the EasyOPD `simple` sidecar and stuffs them into
#        `non_tensor_batch` as variable-length numpy object arrays:
#            - teacher_hidden_states   (object)  np.ndarray[N_i, H]
#            - teacher_input_ids       (object)  np.ndarray[T_i]
#            - teacher_loss_mask       (object)  np.ndarray[T_i] bool
#            - student_response_text   (object)  decoded student responses
#
#     2. The student worker forwards a normal verl mini-batch through
#        FSDP/Megatron and arrives here for distillation loss.
#
#     3. We project hidden states to overlap logits using a
#        process-singleton frozen `lm_head` loaded by `teacher_lm_head.py`
#        (one copy per Python process, shared across micro-batches).
#
#     4. We align student/teacher response tokens via
#        `align_sequences()` and accumulate KL on the overlap sub-vocab.
#
# Contract: matches verl's loss-fn contract:
#     fn(config, distillation_config, model_output, data)
#         -> (distillation_losses, metrics_dict)
# where `distillation_losses` is `[B, resp_len]` zeros at unaligned/non-
# response positions.

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import align_sequences, find_overlap_tokens

logger = logging.getLogger(__name__)


__all__ = [
    "compute_distillation_loss_simple_cross_tokenizer",
    "register_simple_loss",
]


# ---------------------------------------------------------------------------
# Process-level singletons: teacher lm_head + tokenizers + overlap ids.
#
# These are loaded lazily on the FIRST call to the loss function, on the
# device of the incoming model output. All subsequent calls reuse them.
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
            # Re-locate to the new device on demand (tensor parallel may
            # invoke loss fn from a different rank's GPU).
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
# Helpers
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


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_distillation_loss_simple_cross_tokenizer(
    config,
    distillation_config,
    model_output: dict,
    data,
) -> Tuple[torch.Tensor, dict]:
    """Cross-tokenizer KD loss matching verl's loss-fn contract.

    Required student-side keys in `model_output`:
        - "logits": [B, S, V_stu]  (student logits at all positions)

    Required keys in `data` (TensorDict):
        - "response_mask":     [B, S] bool/int — student response mask.
        - "responses":         [B, resp_len] — student response token ids.
        - "prompts":           [B, prompt_len] — student prompt ids.

    Required non-tensor keys in `data` (numpy object arrays of length B):
        - "teacher_hidden_states":  np.ndarray[N_i, H] per sample
        - "teacher_input_ids":      np.ndarray[T_i]
        - "teacher_loss_mask":      np.ndarray[T_i] bool

    Returns:
        distillation_losses: [B, S] (zeros at unaligned / non-response positions)
        metrics: dict with `distillation/align_ratio`,
                 `distillation/overlap_vocab_size`.
    """
    response_mask: torch.Tensor = data["response_mask"]
    if response_mask.is_nested:
        response_mask_dense = response_mask.bool().to_padded_tensor(False)
    else:
        response_mask_dense = response_mask.bool()

    student_logits: torch.Tensor = model_output["logits"]
    bsz, seq_len = student_logits.shape[:2]
    device = student_logits.device
    dtype = student_logits.dtype

    student_input_ids = data["input_ids"] if "input_ids" in data.keys() else None
    if student_input_ids is None:
        # verl fallback: concat prompts + responses.
        student_input_ids = torch.cat([data["prompts"], data["responses"]], dim=1)

    # Pull non-tensor batch entries. They are stored on `data.non_tensor_batch`
    # in DataProto; verl's downstream logic typically merges them into `data`
    # as a dict-like attribute, but to be safe we read from both places.
    nt = getattr(data, "non_tensor_batch", None) or {}
    teacher_hidden_states_arr = nt.get("teacher_hidden_states", None)
    teacher_input_ids_arr = nt.get("teacher_input_ids", None)
    teacher_loss_mask_arr = nt.get("teacher_loss_mask", None)
    if teacher_hidden_states_arr is None:
        # Allow callers that placed the arrays directly into `data` (TensorDict
        # with object-typed entries is unusual but possible during testing).
        teacher_hidden_states_arr = data.get("teacher_hidden_states")
        teacher_input_ids_arr = data.get("teacher_input_ids")
        teacher_loss_mask_arr = data.get("teacher_loss_mask")

    if teacher_hidden_states_arr is None:
        raise KeyError(
            "[EasyOPD:simple] teacher_hidden_states not found in data; "
            "did the agent_loop sidecar populate non_tensor_batch?"
        )

    # Unwrap verl's NonTensorData wrappers if present (TensorDict stores
    # non-tensor batch values via NonTensorData, which exposes `.data`).
    from verl.utils.tensordict_utils import unwrap_non_tensor_data

    teacher_hidden_states_arr = unwrap_non_tensor_data(teacher_hidden_states_arr)
    teacher_input_ids_arr = unwrap_non_tensor_data(teacher_input_ids_arr)
    teacher_loss_mask_arr = unwrap_non_tensor_data(teacher_loss_mask_arr)

    # Resolve student tokenizer path. ActorConfig does not carry the
    # model path directly, so we rely on a process-level environment
    # variable that the student worker sets in `engine_workers.py` from
    # `self.config.model.path` at init time.
    import os as _os

    student_path = _os.environ.get("EASYOPD_STUDENT_MODEL_PATH")
    if student_path is None:
        # Fall back to attribute lookups for unit-test contexts.
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

    loss_config = distillation_config.distillation_loss
    direction = getattr(loss_config, "cross_tokenizer_kl_direction", "forward")

    distillation_losses = torch.zeros(
        (bsz, seq_len), dtype=student_logits.dtype, device=device
    )

    total_aligned = 0
    total_response = 0

    for b in range(bsz):
        # Student-side response positions.
        stu_resp_pos = response_mask_dense[b].nonzero(as_tuple=False).squeeze(-1)
        stu_resp_len = int(stu_resp_pos.numel())
        total_response += stu_resp_len
        if stu_resp_len == 0:
            continue

        # Teacher hidden states are pre-cropped by the sidecar to the
        # response positions only (using `loss_mask`), so they ARE the
        # teacher response tokens, length matches `teacher_loss_mask.sum()`.
        tea_resp_hs_np: np.ndarray = teacher_hidden_states_arr[b]
        if tea_resp_hs_np is None or tea_resp_hs_np.shape[0] == 0:
            continue
        tea_input_ids_np: np.ndarray = teacher_input_ids_arr[b]
        tea_loss_mask_np: np.ndarray = teacher_loss_mask_arr[b].astype(bool)
        # Teacher response token ids (length-aligned with hidden states).
        tea_resp_ids = tea_input_ids_np[tea_loss_mask_np].tolist()

        # Student response token ids.
        stu_resp_ids = student_input_ids[b][stu_resp_pos].detach().cpu().tolist()

        # Decode tokens for greedy character-level alignment.
        stu_tokens = student_tokenizer.convert_ids_to_tokens(stu_resp_ids)
        tea_tokens = teacher_tokenizer.convert_ids_to_tokens(tea_resp_ids)

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

        # Convert local indices into:
        #   * absolute student positions in [seq_len]
        #   * teacher hidden-state row indices (already 0..N-1 in tea_resp_hs)
        stu_abs = stu_resp_pos.index_select(dim=0, index=stu_local)

        # Project teacher hidden states to logits, column-crop to overlap.
        tea_hs = torch.from_numpy(tea_resp_hs_np).to(device=device, dtype=dtype)
        tea_hs_aligned = tea_hs.index_select(dim=0, index=tea_local)  # [N, H]
        with torch.no_grad():
            tea_full_logits = teacher_lm_head(tea_hs_aligned)         # [N, V_tea]
        tea_logits_overlap = tea_full_logits.index_select(
            dim=-1, index=teacher_overlap_ids
        )  # [N, K]

        # Student: gather, then column-crop.
        stu_logits_full = student_logits[b].index_select(dim=0, index=stu_abs)
        stu_logits_overlap = stu_logits_full.index_select(
            dim=-1, index=student_overlap_ids
        )

        if stu_logits_overlap.shape != tea_logits_overlap.shape:
            raise RuntimeError(
                "Aligned student/teacher overlap-logits shape mismatch: "
                f"student={tuple(stu_logits_overlap.shape)} vs "
                f"teacher={tuple(tea_logits_overlap.shape)}"
            )

        kl_per_pos = _kl_div(stu_logits_overlap, tea_logits_overlap, direction)
        distillation_losses[b].index_copy_(
            0, stu_abs, kl_per_pos.to(distillation_losses.dtype)
        )

    align_ratio = float(total_aligned) / max(total_response, 1)

    if align_ratio < 0.5:
        logger.warning(
            "[EasyOPD:simple] align_ratio=%.3f below 0.5; KD signal may be weak.",
            align_ratio,
        )

    from verl.utils.metric import AggregationType, Metric

    metrics: dict = {
        "distillation/align_ratio": Metric(
            AggregationType.MEAN, torch.tensor(align_ratio)
        ),
        "distillation/overlap_vocab_size": Metric(
            AggregationType.MEAN, torch.tensor(float(student_overlap_ids.numel()))
        ),
    }
    return distillation_losses, metrics