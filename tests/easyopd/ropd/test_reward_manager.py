from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from easyopd.methods.ropd.judge.schema import StructuredRubric, RubricCriterion, VerifierScore
from easyopd.methods.ropd.pipeline import ROPDGroup, ROPDPairResult, ROPDRollout
from easyopd.methods.ropd.reward_manager import ROPDRewardManager


class _StaticRubric(StructuredRubric):
    pass


class _StaticTeacherClient:
    def generate(self, raw_prompt: Any, *, uid: str | None = None) -> str:
        del raw_prompt, uid
        return "teacher response"


class _StaticRubricClient:
    def generate(
        self,
        raw_prompt: Any,
        teacher_response: str,
        student_response: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
    ) -> StructuredRubric:
        del raw_prompt, teacher_response, student_response, uid, pair_index
        return StructuredRubric(
            schema_version="ropd.rubric.v1",
            rubrics=[
                RubricCriterion(criterion_id="c1", category="Task", criterion="criterion 1", points=1),
                RubricCriterion(criterion_id="c2", category="Task", criterion="criterion 2", points=1),
            ],
            maximum_score=2,
        )


class _StaticVerifierClient:
    def __init__(self, score: float) -> None:
        self.score = score

    def score(
        self,
        raw_prompt: Any,
        rubric: Any,
        answer: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
        subject: str | None = None,
    ) -> VerifierScore:
        del raw_prompt, answer, uid, pair_index, subject
        return VerifierScore(
            schema_version="ropd.verifier.v1",
            judgement=[True, True],
            final_score=float(self.score),
        )


def _make_pipeline(teacher_score: float, student_score: float):
    from easyopd.methods.ropd.pipeline import ROPDPipeline

    class _SwitchingVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, raw_prompt, rubric, answer, *, uid=None, pair_index=None, subject=None):
            del raw_prompt, rubric, answer, uid, pair_index
            value = teacher_score if subject == "teacher" else student_score
            return VerifierScore(
                schema_version="ropd.verifier.v1",
                judgement=[True, True],
                final_score=float(value),
            )

    return ROPDPipeline(
        teacher_client=_StaticTeacherClient(),
        rubric_client=_StaticRubricClient(),
        verifier_client=_SwitchingVerifier(),
        max_pair_concurrency=1,
        max_verifier_subject_concurrency=1,
    )


def test_ropd_reward_manager_constructor_accepts_ropd_kwarg() -> None:
    manager = ROPDRewardManager(
        tokenizer=None,
        num_examine=0,
        compute_score=None,
        pipeline=_make_pipeline(2.0, 1.0),
    )
    assert manager.pipeline is not None


def test_ropd_pipeline_produces_normalized_reward() -> None:
    pipeline = _make_pipeline(teacher_score=2.0, student_score=1.0)
    rollout = ROPDRollout(batch_index=0, response_text="student answer", response_length=2)
    group = ROPDGroup(uid="sample", raw_prompt="Question?", rollouts=(rollout,))

    pair_results = pipeline.evaluate_group(group)
    assert len(pair_results) == 1
    assert isinstance(pair_results[0], ROPDPairResult)
    assert pair_results[0].teacher_score == pytest.approx(2.0)
    assert pair_results[0].student_score == pytest.approx(1.0)
    # reward = (student - teacher) / rubric.maximum_score == -0.5
    assert pair_results[0].reward == pytest.approx(-0.5)


def test_ropd_reward_manager_extra_info_includes_required_keys() -> None:
    defaults = ROPDRewardManager.EXTRA_INFO_DEFAULTS
    for key in (
        "pair_index",
        "group_size",
        "teacher_score",
        "student_score",
        "reward_gap",
        "student_win",
        "fallback_used",
        "judge_error",
    ):
        assert key in defaults, key
