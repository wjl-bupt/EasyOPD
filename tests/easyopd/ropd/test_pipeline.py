from __future__ import annotations

from easyopd.methods.ropd.pipeline import (
    Group,
    ROPDGroup,
    ROPDPairResult,
    ROPDPipeline,
    ROPDRollout,
    Rollout,
    canonicalize_raw_prompt,
    normalize_raw_prompt,
)


def test_normalize_raw_prompt_round_trips_messages() -> None:
    raw_prompt = [{"content": "Question?", "role": "user"}]

    assert normalize_raw_prompt(raw_prompt) == ({"content": "Question?", "role": "user"},)
    assert canonicalize_raw_prompt(raw_prompt) == '[{"content": "Question?", "role": "user"}]'


def test_normalize_raw_prompt_accepts_string() -> None:
    assert normalize_raw_prompt("hello") == "hello"
    assert canonicalize_raw_prompt("hello") == '"hello"'


def test_ropd_pipeline_types_extend_neutral_types() -> None:
    rollout = ROPDRollout(batch_index=0, response_text="answer", response_length=1)
    group = ROPDGroup(uid="sample", raw_prompt="Question?", rollouts=(rollout,))

    assert isinstance(rollout, Rollout)
    assert isinstance(group, Group)


def test_ropd_pair_result_default_fields() -> None:
    pair = ROPDPairResult(
        batch_index=0,
        pair_index=0,
        teacher_score=1.0,
        student_score=0.5,
        reward=-0.5,
    )

    assert pair.student_win is False
    assert pair.fallback_used is False
    assert pair.judge_error is False
    assert pair.error_details == {}


def test_ropd_pipeline_class_is_exposed() -> None:
    assert ROPDPipeline.__name__ == "ROPDPipeline"
