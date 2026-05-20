# Copyright 2026 EasyOPD Contributors
#
# Unit tests for `easyopd.methods.simple.alignment`.
#
# These tests intentionally use lightweight stub tokenizers so they can run
# without HuggingFace model downloads. End-to-end correctness on real
# tokenizers (Qwen + Llama) is covered by the integration tests in task 8.

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import pytest

from easyopd.methods.simple.alignment import (
    align_sequences,
    find_overlap_tokens,
    normalize_token_key,
)


class _StubTokenizer:
    """A minimal HF-tokenizer-shaped stub for unit tests."""

    def __init__(self, vocab: dict[str, int], eos_token_id: int, eos_token: str):
        self._vocab = vocab
        self.eos_token_id = eos_token_id
        self.eos_token = eos_token

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


# ---------------------------------------------------------------------------
# normalize_token_key
# ---------------------------------------------------------------------------

def test_normalize_token_key_replaces_g_marker():
    assert normalize_token_key("\u0120hello") == "\u2581hello"
    assert normalize_token_key("\u2581hello") == "\u2581hello"
    assert normalize_token_key("hello") == "hello"


# ---------------------------------------------------------------------------
# find_overlap_tokens
# ---------------------------------------------------------------------------

def test_find_overlap_tokens_basic_intersection():
    student = _StubTokenizer(
        vocab={"\u2581the": 1, "\u2581cat": 2, "\u2581dog": 3, "</s>": 100},
        eos_token_id=100,
        eos_token="</s>",
    )
    teacher = _StubTokenizer(
        vocab={"\u0120the": 11, "\u0120cat": 12, "\u0120fox": 13, "</s>": 200},
        eos_token_id=200,
        eos_token="</s>",
    )
    s_ids, t_ids = find_overlap_tokens(student, teacher)

    # Build mapping from normalized key for an order-independent check.
    s_vocab_norm = {normalize_token_key(k): v for k, v in student.get_vocab().items()}
    t_vocab_norm = {normalize_token_key(k): v for k, v in teacher.get_vocab().items()}
    expected_keys = {"\u2581the", "\u2581cat", "</s>"}

    pairs = set(zip(s_ids, t_ids))
    expected_pairs = {(s_vocab_norm[k], t_vocab_norm[k]) for k in expected_keys}

    # </s> is in both vocabs already, so EOS fallback should NOT trigger an
    # extra append (KDFlow behaviour: condition is `not in either`).
    assert pairs == expected_pairs
    assert len(s_ids) == len(t_ids) == 3


def test_find_overlap_tokens_eos_fallback_appends_when_missing():
    # Student EOS is 100 but the only "</s>" token id in the overlap is on
    # neither side because they don't share that key.
    student = _StubTokenizer(
        vocab={"\u2581a": 1, "<eos_stu>": 100},
        eos_token_id=100,
        eos_token="<eos_stu>",
    )
    teacher = _StubTokenizer(
        vocab={"\u2581a": 11, "<eos_tea>": 200},
        eos_token_id=200,
        eos_token="<eos_tea>",
    )
    s_ids, t_ids = find_overlap_tokens(student, teacher)

    # Overlap key "\u2581a" gives one pair; EOS fallback appends one more pair.
    assert len(s_ids) == len(t_ids) == 2
    assert 100 in s_ids
    assert 200 in t_ids


def test_find_overlap_tokens_empty_overlap_still_appends_eos():
    student = _StubTokenizer(
        vocab={"\u2581foo": 1, "<eos_s>": 100},
        eos_token_id=100,
        eos_token="<eos_s>",
    )
    teacher = _StubTokenizer(
        vocab={"\u2581bar": 11, "<eos_t>": 200},
        eos_token_id=200,
        eos_token="<eos_t>",
    )
    s_ids, t_ids = find_overlap_tokens(student, teacher)
    assert s_ids == [100]
    assert t_ids == [200]


# ---------------------------------------------------------------------------
# align_sequences
# ---------------------------------------------------------------------------

def test_align_sequences_identity_on_same_tokenizer():
    seq = ["\u2581hello", "\u2581world", "</s>"]
    t_idx, s_idx = align_sequences(seq, seq, "</s>", "</s>")
    assert t_idx == [0, 1, 2]
    assert s_idx == [0, 1, 2]


def test_align_sequences_cross_tokenizer_shared_prefix_token():
    # Both tokenizers emit "▁the" as the first token but split the rest
    # differently. The greedy walker matches the shared first token and
    # then advances independently.
    teacher = ["\u2581the", "\u2581cat"]            # cumulative: "the", "thecat"
    student = ["\u2581the", "\u2581ca", "t"]        # cumulative: "the", "theca", "thecat"
    t_idx, s_idx = align_sequences(teacher, student, "</s>", "</s>")
    assert t_idx == [0]
    assert s_idx == [0]


def test_align_sequences_eos_pair_matches_even_if_strings_differ():
    teacher = ["\u2581a", "</s_t>"]
    student = ["\u2581a", "</s_s>"]
    t_idx, s_idx = align_sequences(teacher, student, "</s_t>", "</s_s>")
    # First pair: "a" matches; second pair: EOS-pair shortcut allows match
    # even though the stripped strings differ.
    assert (t_idx, s_idx) == ([0, 1], [0, 1])


def test_align_sequences_completely_unalignable_returns_partial_or_empty():
    teacher = ["\u2581abc"]
    student = ["\u2581xyz"]
    t_idx, s_idx = align_sequences(teacher, student, "</s>", "</s>")
    # Histories diverge immediately; greedy walker advances both and exits.
    # The contract is "no errors, never hallucinate alignments" — so we
    # only assert that no spurious aligned pairs are produced.
    assert len(t_idx) == len(s_idx)
    assert t_idx == []
    assert s_idx == []


def test_align_sequences_contract_lengths_equal():
    """The two returned index lists must always have equal length."""
    cases = [
        (["\u2581foo", "bar"], ["\u2581foob", "ar"]),
        (["\u2581a", "\u2581b"], ["\u2581a", "\u2581c"]),
        ([], []),
    ]
    for tea, stu in cases:
        t, s = align_sequences(tea, stu, "</s>", "</s>")
        assert len(t) == len(s)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
