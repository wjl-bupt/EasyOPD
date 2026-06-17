# Copyright 2026 EasyOPD Contributors
#
# Shared greedy character-level token-sequence alignment used by ULD and
# DSKD-eta. Equivalent to KDFlow's `_align_sequences` (uld.py / dskd.py):
# normalize SentencePiece (`▁`) and byte-level BPE (`Ġ`) prefixes, then
# greedily walk both sequences keeping cumulative-text invariants in sync.

from __future__ import annotations

from typing import Sequence


__all__ = ["align_token_sequences", "EOS_TOKEN_MARKER"]


EOS_TOKEN_MARKER = "<|eos|>"


def _normalize(token: str) -> str:
    """Strip whitespace prefixes used by SentencePiece (▁) and byte-level BPE (Ġ)."""
    return token.replace("\u2581", "").replace("\u0120", "")


def align_token_sequences(
    teacher_tokens: Sequence[str],
    student_tokens: Sequence[str],
    teacher_eos_token: str | None = None,
    student_eos_token: str | None = None,
) -> tuple[list[int], list[int]]:
    """Greedy character-level alignment between two token sequences.

    Args:
        teacher_tokens: List of teacher tokens (e.g. ``convert_ids_to_tokens(...)``).
        student_tokens: List of student tokens.
        teacher_eos_token: Optional EOS token literal for the teacher tokenizer
            (mapped to a canonical marker so alignment is robust to differing
            EOS surface forms).
        student_eos_token: Same for the student tokenizer.

    Returns:
        ``(teacher_aligned_idx, student_aligned_idx)`` — two equal-length lists
        of strictly-increasing indices selecting the aligned positions on each
        side. If both sequences are token-identical (after EOS / prefix
        normalization), the trivial alignment ``range(N)`` is returned.
    """
    tea_seq = [
        EOS_TOKEN_MARKER if (teacher_eos_token is not None and tok == teacher_eos_token) else _normalize(tok)
        for tok in teacher_tokens
    ]
    stu_seq = [
        EOS_TOKEN_MARKER if (student_eos_token is not None and tok == student_eos_token) else _normalize(tok)
        for tok in student_tokens
    ]

    if tea_seq == stu_seq:
        indices = list(range(len(tea_seq)))
        return indices, indices

    i, j = 0, 0
    t_align: list[int] = []
    s_align: list[int] = []
    history_tea = ""
    history_stu = ""

    while i < len(tea_seq) and j < len(stu_seq):
        if history_tea == history_stu and tea_seq[i] == stu_seq[j]:
            common = tea_seq[i]
            history_tea += common
            history_stu += common
            t_align.append(i)
            s_align.append(j)
            i += 1
            j += 1
        elif len(history_tea) > len(history_stu):
            history_stu += stu_seq[j]
            j += 1
        elif len(history_tea) < len(history_stu):
            history_tea += tea_seq[i]
            i += 1
        else:
            history_tea += tea_seq[i]
            history_stu += stu_seq[j]
            i += 1
            j += 1

    return t_align, s_align
