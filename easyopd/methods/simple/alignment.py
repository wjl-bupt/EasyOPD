# Copyright 2026 EasyOPD Contributors
#
# Pure-algorithm utilities for cross-tokenizer alignment, ported (and made
# verl-friendly) from KDFlow's `simple_ctkd._find_overlap_tokens` and
# `_align_sequences`.
#
# These helpers have no verl / torch / transformers dependency at *import*
# time. `find_overlap_tokens` only requires objects that quack like
# HuggingFace tokenizers (`get_vocab()` + `eos_token_id`). `align_sequences`
# operates on Python lists of token strings.
#
# Numerical equivalence with KDFlow is required (see tests).

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "normalize_token_key",
    "find_overlap_tokens",
    "align_sequences",
]


# ---------------------------------------------------------------------------
# Overlap discovery
# ---------------------------------------------------------------------------

def normalize_token_key(key: str) -> str:
    """Normalize tokenizer vocab keys so BPE-style (`Ġ`) and SentencePiece
    (`▁`) word-boundary markers compare equal.

    KDFlow does the same single substitution: `Ġ` -> `▁`.
    """
    return key.replace("\u0120", "\u2581")  # Ġ -> ▁


def find_overlap_tokens(
    student_tokenizer,
    teacher_tokenizer,
) -> Tuple[List[int], List[int]]:
    """Find the overlap token ids between two HuggingFace tokenizers.

    Reproduces KDFlow `simple_ctkd._find_overlap_tokens` semantics:

    * Build student/teacher vocabs with keys normalized via `normalize_token_key`.
      Note that this normalization is *lossy*: two distinct raw keys (e.g.
      "Ġhello" and "▁hello") collapse onto the same normalized key, and the
      last write to the dict wins. This matches KDFlow's dict-comprehension
      behaviour exactly.
    * Take the set intersection of normalized keys.
    * For each overlap key, look up its id in each side. The iteration order
      of a Python `set` is not deterministic across runs, so the *order* of
      the returned ids is unspecified — but the two lists are aligned (the
      i-th student id and i-th teacher id describe the same normalized
      token).
    * Append the EOS id on each side as a fallback if either EOS is missing
      from its respective overlap list. (KDFlow uses an `or`, so a single
      side missing triggers the append on both sides.)

    Args:
        student_tokenizer: HuggingFace-compatible tokenizer with
            `get_vocab()` and `eos_token_id`.
        teacher_tokenizer: ditto.

    Returns:
        (student_overlap_ids, teacher_overlap_ids) — two equal-length Python
        lists of ints, with student[i] / teacher[i] aligned.
    """
    student_vocab = {
        normalize_token_key(k): v for k, v in student_tokenizer.get_vocab().items()
    }
    teacher_vocab = {
        normalize_token_key(k): v for k, v in teacher_tokenizer.get_vocab().items()
    }
    overlap_keys = set(student_vocab.keys()) & set(teacher_vocab.keys())

    student_ids: List[int] = [student_vocab[k] for k in overlap_keys]
    teacher_ids: List[int] = [teacher_vocab[k] for k in overlap_keys]

    stu_eos = student_tokenizer.eos_token_id
    tea_eos = teacher_tokenizer.eos_token_id
    if stu_eos is not None and tea_eos is not None:
        if stu_eos not in student_ids or tea_eos not in teacher_ids:
            student_ids.append(stu_eos)
            teacher_ids.append(tea_eos)

    return student_ids, teacher_ids


# ---------------------------------------------------------------------------
# Sequence alignment
# ---------------------------------------------------------------------------

def _strip_word_marker(token: str) -> str:
    """Strip leading/embedded `▁` and `Ġ` word-boundary markers, matching
    KDFlow's per-token preprocessing for alignment.
    """
    return token.replace("\u2581", "").replace("\u0120", "")


def align_sequences(
    teacher_tokens: Sequence[str],
    student_tokens: Sequence[str],
    teacher_eos_token: str | None = None,
    student_eos_token: str | None = None,
) -> Tuple[List[int], List[int]]:
    """Character-level greedy alignment between teacher and student token
    sequences. Numerically equivalent to KDFlow `_align_sequences`.

    Args:
        teacher_tokens: list of teacher-side token strings (e.g. output of
            `tokenizer.convert_ids_to_tokens(ids)`).
        student_tokens: ditto on the student side.
        teacher_eos_token: teacher's EOS *string* (post-strip, the comparison
            uses the post-strip form). If `None`, EOS-pair short-circuit is
            disabled.
        student_eos_token: ditto on the student side.

    Returns:
        (teacher_aligned_idx, student_aligned_idx) — two equal-length Python
        lists of ints. `teacher_tokens[teacher_aligned_idx[i]]` and
        `student_tokens[student_aligned_idx[i]]` are character-aligned.
    """
    # Strip word-boundary markers (operate on copies; do NOT mutate inputs).
    tea_seq = [_strip_word_marker(t) for t in teacher_tokens]
    stu_seq = [_strip_word_marker(t) for t in student_tokens]

    # Fast path: if the stripped sequences are exactly equal, alignment is
    # the identity. This is the common case when student and teacher share
    # a tokenizer.
    if tea_seq == stu_seq:
        idx = list(range(len(tea_seq)))
        return idx, idx

    tea_eos_stripped = (
        _strip_word_marker(teacher_eos_token) if teacher_eos_token is not None else None
    )
    stu_eos_stripped = (
        _strip_word_marker(student_eos_token) if student_eos_token is not None else None
    )

    i = 0
    j = 0
    t_align: List[int] = []
    s_align: List[int] = []
    history_tea = ""
    history_stu = ""

    while i < len(tea_seq) and j < len(stu_seq):
        is_eos_match = (
            tea_eos_stripped is not None
            and stu_eos_stripped is not None
            and tea_seq[i] == tea_eos_stripped
            and stu_seq[j] == stu_eos_stripped
        )
        if history_tea == history_stu and (tea_seq[i] == stu_seq[j] or is_eos_match):
            common_text = tea_seq[i]
            history_tea += common_text
            history_stu += common_text
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
            # Equal length but content differs — advance both to keep
            # progress (matches KDFlow behaviour).
            history_tea += tea_seq[i]
            history_stu += stu_seq[j]
            i += 1
            j += 1

    return t_align, s_align