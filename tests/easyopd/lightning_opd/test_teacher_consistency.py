"""Tests for teacher consistency check."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest


def test_consistent_teachers():
    """Same teacher IDs should return True."""
    from easyopd.methods.lightning_opd.teacher_consistency import check_teacher_consistency

    assert check_teacher_consistency("Qwen/Qwen3-8B", "Qwen/Qwen3-8B") is True


def test_inconsistent_teachers_raises():
    """Different teacher IDs should raise LightningOPDTeacherInconsistency."""
    from easyopd.methods.lightning_opd.teacher_consistency import (
        LightningOPDTeacherInconsistency,
        check_teacher_consistency,
    )

    with pytest.raises(LightningOPDTeacherInconsistency):
        check_teacher_consistency("Qwen/Qwen3-8B", "Qwen/Qwen3-32B")


def test_allow_mismatch():
    """With allow_mismatch=True, should return False instead of raising."""
    from easyopd.methods.lightning_opd.teacher_consistency import check_teacher_consistency

    result = check_teacher_consistency(
        "Qwen/Qwen3-8B", "Qwen/Qwen3-32B", allow_mismatch=True
    )
    assert result is False


def test_whitespace_stripping():
    """Whitespace in IDs should be stripped for comparison."""
    from easyopd.methods.lightning_opd.teacher_consistency import check_teacher_consistency

    assert check_teacher_consistency("  Qwen/Qwen3-8B  ", "Qwen/Qwen3-8B") is True


def test_exception_message():
    """Exception message should include both teacher IDs."""
    from easyopd.methods.lightning_opd.teacher_consistency import (
        LightningOPDTeacherInconsistency,
        check_teacher_consistency,
    )

    with pytest.raises(LightningOPDTeacherInconsistency, match="Qwen/Qwen3-8B"):
        check_teacher_consistency("Qwen/Qwen3-8B", "Qwen/Qwen3-32B")
