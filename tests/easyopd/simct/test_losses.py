# Copyright 2026 EasyOPD Contributors
#
# Unit tests for `easyopd.methods.simct.losses`.
#
# These tests use lightweight tokenizer stubs and synthetic logits so they do
# not require model downloads, GPUs, SGLang, or Ray.

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import torch

from easyopd.methods.simct.losses import (
    EOS_MARKER,
    align_label_ids_with_spans,
    build_virtual_vocab_logits,
    decode_token_texts,
    register_simct_loss,
)
from verl.trainer.distillation.losses import (
    DISTILLATION_LOSS_REGISTRY,
    DISTILLATION_SETTINGS_REGISTRY,
)


class _DecodeTokenizer:
    def __init__(self, pieces, eos_token_id=99):
        self.pieces = dict(pieces)
        self.eos_token_id = eos_token_id
        self.eos_token = "<eos>"
        self.all_special_ids = [eos_token_id, 1000]

    def decode(self, ids, skip_special_tokens=False):
        assert len(ids) == 1
        return self.pieces[int(ids[0])]


def test_decode_token_texts_uses_decode_and_preserves_newline():
    tokenizer = _DecodeTokenizer({1: "hello", 2: "\n", 3: " world"})
    assert decode_token_texts([1, 2, 3], tokenizer) == ["hello", "\n", " world"]
    assert decode_token_texts([99], tokenizer) == [EOS_MARKER]
    assert decode_token_texts([1000], tokenizer) == [""]


def test_align_label_ids_with_spans_identifies_cross_tokenizer_span():
    teacher = _DecodeTokenizer({10: "ab", 11: "c"})
    student = _DecodeTokenizer({20: "a", 21: "bc"})

    segments, teacher_ids, student_ids = align_label_ids_with_spans(
        teacher_label_ids=[10, 11],
        student_label_ids=[20, 21],
        teacher_tokenizer=teacher,
        student_tokenizer=student,
    )

    assert teacher_ids == [10, 11]
    assert student_ids == [20, 21]
    assert segments == [(0, 2, 0, 2)]


def test_align_label_ids_with_spans_handles_one_to_one_and_span_mix():
    teacher = _DecodeTokenizer({10: "x", 11: "ab", 12: "c"})
    student = _DecodeTokenizer({20: "x", 21: "a", 22: "bc"})

    segments, _, _ = align_label_ids_with_spans(
        teacher_label_ids=[10, 11, 12],
        student_label_ids=[20, 21, 22],
        teacher_tokenizer=teacher,
        student_tokenizer=student,
    )

    assert segments == [(0, 1, 0, 1), (1, 3, 1, 3)]


def test_align_label_ids_with_spans_unalignable_returns_empty():
    teacher = _DecodeTokenizer({10: "abc"})
    student = _DecodeTokenizer({20: "xyz"})

    segments, _, _ = align_label_ids_with_spans(
        teacher_label_ids=[10],
        student_label_ids=[20],
        teacher_tokenizer=teacher,
        student_tokenizer=student,
    )

    assert segments == []


def test_build_virtual_vocab_logits_span_dim_with_sum_and_masking():
    """Span segments get an extra span dimension (sum of self-logits) and
    the first token's overlap position is masked to -1e9."""
    segments = [(0, 2, 0, 2)]
    student_logits = torch.zeros(2, 8)
    teacher_logits = torch.zeros(2, 9)
    # student: overlap_ids=[1,2], label_ids=[4,5]
    # student_logits[0, 1]=1.0 (overlap pos 0), student_logits[0, 2]=2.0 (overlap pos 1)
    # student_logits[0, 4]=4.0 (self-logit for first token id=4)
    # student_logits[1, 5]=6.0 (self-logit for second token id=5)
    student_logits[0, 1] = 1.0
    student_logits[0, 2] = 2.0
    student_logits[0, 4] = 4.0
    student_logits[1, 5] = 6.0
    # teacher: overlap_ids=[3,4], label_ids=[6,7]
    # teacher_logits[0, 3]=3.0 (overlap pos 0), teacher_logits[0, 4]=5.0 (overlap pos 1)
    # teacher_logits[0, 6]=8.0 (self-logit for first token id=6)
    # teacher_logits[1, 7]=10.0 (self-logit for second token id=7)
    teacher_logits[0, 3] = 3.0
    teacher_logits[0, 4] = 5.0
    teacher_logits[0, 6] = 8.0
    teacher_logits[1, 7] = 10.0

    student_overlap_ids = torch.tensor([1, 2], dtype=torch.long)
    teacher_overlap_ids = torch.tensor([3, 4], dtype=torch.long)

    student_virtual, teacher_virtual, is_span = build_virtual_vocab_logits(
        segments=segments,
        student_logits_aligned=student_logits,
        teacher_logits_aligned=teacher_logits,
        student_label_ids=[4, 5],
        teacher_label_ids=[6, 7],
        student_overlap_ids=student_overlap_ids,
        teacher_overlap_ids=teacher_overlap_ids,
    )

    # Shape: num_overlap(2) + num_spans(1) = 3
    assert student_virtual.shape == teacher_virtual.shape == (1, 3)
    assert is_span == [True]

    # First-token masking: student label_ids[0]=4 is NOT in overlap_ids=[1,2],
    # so no masking happens on student overlap. Teacher label_ids[0]=6 is NOT
    # in teacher overlap_ids=[3,4], so no masking on teacher overlap either.
    # Overlap values pass through unchanged.
    assert torch.allclose(student_virtual[0, :2], torch.tensor([1.0, 2.0]))
    assert torch.allclose(teacher_virtual[0, :2], torch.tensor([3.0, 5.0]))

    # Span dim = MEAN of self-logits: student (4.0+6.0)/2=5.0, teacher (8.0+10.0)/2=9.0
    assert torch.allclose(student_virtual[0, 2], torch.tensor(5.0))
    assert torch.allclose(teacher_virtual[0, 2], torch.tensor(9.0))


def test_build_virtual_vocab_logits_first_token_masking():
    """When first token IS in overlap, its overlap position is masked to -1e9."""
    segments = [(0, 2, 0, 2)]
    student_logits = torch.zeros(2, 8)
    teacher_logits = torch.zeros(2, 9)
    # student: overlap_ids=[1,2], label_ids=[1, 5]  (first token id=1 IS in overlap at pos 0)
    student_logits[0, 1] = 7.0  # overlap pos 0 — will be masked
    student_logits[0, 2] = 8.0  # overlap pos 1 — kept
    student_logits[0, 1] = 7.0  # self-logit for first token (id=1)
    student_logits[1, 5] = 6.0  # self-logit for second token (id=5)
    # teacher: overlap_ids=[3,4], label_ids=[3, 7]  (first token id=3 IS in overlap at pos 0)
    teacher_logits[0, 3] = 9.0  # overlap pos 0 — will be masked
    teacher_logits[0, 4] = 5.0  # overlap pos 1 — kept
    teacher_logits[0, 3] = 9.0  # self-logit for first token (id=3)
    teacher_logits[1, 7] = 10.0  # self-logit for second token (id=7)

    student_virtual, teacher_virtual, is_span = build_virtual_vocab_logits(
        segments=segments,
        student_logits_aligned=student_logits,
        teacher_logits_aligned=teacher_logits,
        student_label_ids=[1, 5],
        teacher_label_ids=[3, 7],
        student_overlap_ids=torch.tensor([1, 2], dtype=torch.long),
        teacher_overlap_ids=torch.tensor([3, 4], dtype=torch.long),
    )

    assert student_virtual.shape == (1, 3)  # 2 overlap + 1 span
    assert is_span == [True]
    # First token's overlap position is masked to -1e9
    assert student_virtual[0, 0].item() == -1e9
    assert teacher_virtual[0, 0].item() == -1e9
    # Second overlap position is kept
    assert student_virtual[0, 1].item() == 8.0
    assert teacher_virtual[0, 1].item() == 5.0
    # Span dim = mean: student (7.0+6.0)/2=6.5, teacher (9.0+10.0)/2=9.5
    assert torch.allclose(student_virtual[0, 2], torch.tensor(6.5))
    assert torch.allclose(teacher_virtual[0, 2], torch.tensor(9.5))


def test_build_virtual_vocab_logits_one_to_one_segment():
    """1:1 segments have no span dimension appended."""
    segments = [(0, 1, 0, 1)]
    student_logits = torch.zeros(1, 8)
    teacher_logits = torch.zeros(1, 9)
    student_logits[0, 1] = 7.0
    student_logits[0, 2] = 8.0
    teacher_logits[0, 3] = 11.0
    teacher_logits[0, 4] = 12.0

    student_virtual, teacher_virtual, is_span = build_virtual_vocab_logits(
        segments=segments,
        student_logits_aligned=student_logits,
        teacher_logits_aligned=teacher_logits,
        student_label_ids=[1],
        teacher_label_ids=[3],
        student_overlap_ids=torch.tensor([1, 2], dtype=torch.long),
        teacher_overlap_ids=torch.tensor([3, 4], dtype=torch.long),
    )

    # No span segments → no span dim appended → shape is (1, num_overlap=2)
    assert student_virtual.shape == teacher_virtual.shape == (1, 2)
    assert is_span == [False]
    assert torch.allclose(student_virtual[0], torch.tensor([7.0, 8.0]))
    assert torch.allclose(teacher_virtual[0], torch.tensor([11.0, 12.0]))


def test_register_simct_loss_registers_new_and_legacy_names():
    register_simct_loss()
    assert "simct" in DISTILLATION_LOSS_REGISTRY
    assert "span_ctkd" in DISTILLATION_LOSS_REGISTRY
    assert DISTILLATION_LOSS_REGISTRY["simct"] is DISTILLATION_LOSS_REGISTRY["span_ctkd"]
    assert DISTILLATION_SETTINGS_REGISTRY["simct"].use_cross_tokenizer is True
    assert DISTILLATION_SETTINGS_REGISTRY["span_ctkd"].use_cross_tokenizer is True
