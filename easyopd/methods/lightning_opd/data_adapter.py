"""Data adapter for Lightning-OPD teacher log-probabilities.

Handles converting the ``teacher_log_probs`` column from parquet
(ragged list[float] per sample) into a padded tensor aligned with
the response portion of the batch, ready for the advantage estimator.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LightningOPDLogprobLengthMismatch(ValueError):
    """Raised when teacher_log_probs length does not match response_length."""


def attach_teacher_log_probs(batch) -> None:
    """Convert ``teacher_log_probs`` from non-tensor to tensor in-place.

    Reads ``batch.non_tensor_batch["teacher_log_probs"]`` (a numpy object
    array of ragged ``list[float]``), pads each to the max response length
    in the batch, and stores the result as ``batch.batch["teacher_log_probs"]``
    (a ``(B, T)`` float tensor aligned with ``response_mask``).

    If the column is absent, this is a no-op.

    Args:
        batch: A ``DataProto`` with ``batch`` and ``non_tensor_batch`` attrs.
    """
    ntb = getattr(batch, "non_tensor_batch", None)
    if ntb is None or "teacher_log_probs" not in ntb:
        return

    teacher_lps_raw = ntb["teacher_log_probs"]
    if teacher_lps_raw is None:
        return

    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        logger.warning("response_mask not in batch; cannot align teacher_log_probs")
        return

    B, T = response_mask.shape
    max_resp = int(response_mask.sum(dim=1).max().item())

    padded = torch.full((B, max_resp), float("-inf"), dtype=torch.float32)
    for i in range(B):
        lp_list = teacher_lps_raw[i]
        if lp_list is None:
            continue
        if isinstance(lp_list, np.ndarray):
            lp_list = lp_list.tolist()
        lp = torch.tensor(list(lp_list), dtype=torch.float32)
        length = min(len(lp), max_resp)
        if len(lp) != length:
            logger.debug(
                "teacher_log_probs length %d != response_length %d for sample %d; truncating",
                len(lp), length, i,
            )
        padded[i, :length] = lp[:length]

    batch.batch["teacher_log_probs"] = padded


def register_data_adapter() -> None:
    """Placeholder for future dynamic registration if needed.

    Currently the adapter is called as a hook in ``ray_trainer.py``.
    This function exists so ``__init__.py::register()`` has a stable
    entry point.
    """
    pass
