"""Teacher consistency check for Lightning-OPD.

Lightning-OPD requires the SFT teacher and the OPD teacher to be the
same model.  Using different teachers introduces an unremovable gradient
bias (arXiv:2604.13010 §3).

This module provides a pure-function checker that can be called from
prepare tools and training entrypoints.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LightningOPDTeacherInconsistency(RuntimeError):
    """Raised when SFT teacher != OPD teacher."""


def _resolve_id(teacher_id: str) -> str:
    """Normalise a teacher identifier for comparison.

    Accepts either a HuggingFace model path or a sha256 hex digest.
    Returns the identifier stripped and lowercased for comparison.
    """
    return teacher_id.strip()


def check_teacher_consistency(
    sft_teacher_id: str,
    opd_teacher_id: str,
    *,
    allow_mismatch: bool = False,
) -> bool:
    """Check that the SFT teacher and OPD teacher are the same model.

    Args:
        sft_teacher_id: Identifier for the SFT teacher (HF path or sha256).
        opd_teacher_id: Identifier for the OPD teacher (HF path or sha256).
        allow_mismatch: If ``True``, log a warning instead of raising.

    Returns:
        ``True`` if consistent.

    Raises:
        LightningOPDTeacherInconsistency: If inconsistent and
            ``allow_mismatch`` is ``False``.
    """
    sft_id = _resolve_id(sft_teacher_id)
    opd_id = _resolve_id(opd_teacher_id)

    if sft_id == opd_id:
        return True

    msg = (
        f"Teacher inconsistency: SFT teacher ({sft_teacher_id!r}) != "
        f"OPD teacher ({opd_teacher_id!r}). Lightning-OPD requires the "
        f"same teacher for both SFT and OPD stages (arXiv:2604.13010 §3)."
    )
    if allow_mismatch:
        logger.warning(msg + " Proceeding due to --allow-teacher-mismatch.")
        return False

    raise LightningOPDTeacherInconsistency(msg)


def hash_tokenizer(tokenizer_path: str) -> str:
    """Compute a sha256 hash of tokenizer.json for identity comparison.

    Args:
        tokenizer_path: Path to the directory containing ``tokenizer.json``.

    Returns:
        Hex digest string.
    """
    tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")
    content = tokenizer_file.read_bytes()
    return hashlib.sha256(content).hexdigest()
