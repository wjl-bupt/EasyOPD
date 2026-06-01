from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from easyopd.methods.ropd.judge.schema import (
    JudgeClientError,
    RubricCriterion,
    StructuredRubric,
    VerifierScore,
)
from easyopd.methods.ropd.pipeline import ROPDPipeline
from easyopd.methods.ropd.reward_manager import ROPDRewardManager
from verl.protocol import DataProto

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for ROPD GPU smoke tests"),
]


class _CudaAwareTokenizer:
    def __init__(self) -> None:
        self.decoded_devices: list[torch.device] = []

    def decode(self, response_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        self.decoded_devices.append(response_ids.device)
        token_ids = response_ids.detach().cpu().tolist()
        return " ".join(str(token_id) for token_id in token_ids)


class _RecordingTeacherClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, str | None]] = []

    def generate(self, raw_prompt: Any, *, uid: str | None = None) -> str:
        self.calls.append((raw_prompt, uid))
        return f"teacher answer for {uid}"


class _RecordingRubricClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, str, str, str | None, int | None]] = []

    def generate(
        self,
        raw_prompt: Any,
        teacher_response: str,
        student_response: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
    ) -> StructuredRubric:
        self.calls.append((raw_prompt, teacher_response, student_response, uid, pair_index))
        return StructuredRubric(
            schema_version="ropd.rubric.v1",
            rubrics=[
                RubricCriterion(criterion_id="c1", category="Task", criterion="Correctness", points=1),
                RubricCriterion(criterion_id="c2", category="Task", criterion="Completeness", points=1),
            ],
            maximum_score=2,
        )


class _RecordingVerifierClient:
    def __init__(
        self,
        *,
        student_scores: dict[tuple[str, int], float] | None = None,
        fail_once: set[tuple[str, int, str]] | None = None,
    ) -> None:
        self.student_scores = student_scores or {}
        self.fail_once = set(fail_once or set())
        self.failed_once: set[tuple[str, int, str]] = set()
        self.calls: list[tuple[Any, Any, str, str | None, int | None, str | None]] = []

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
        self.calls.append((raw_prompt, rubric, answer, uid, pair_index, subject))
        failure_key = (str(uid), int(pair_index), str(subject))
        if failure_key in self.fail_once and failure_key not in self.failed_once:
            self.failed_once.add(failure_key)
            raise JudgeClientError(
                stage="verifier",
                error_type="timeout",
                message="transient verifier timeout",
                retriable=True,
            )

        if subject == "teacher":
            final_score = 2.0
        else:
            final_score = self.student_scores.get((str(uid), int(pair_index)), 1.0)
        return VerifierScore(
            schema_version="ropd.verifier.v1",
            judgement=[True, True],
            final_score=final_score,
        )


def _make_cuda_data(
    *,
    responses: list[list[int]],
    attention_mask: list[list[int]],
    raw_prompts: list[str],
    uids: list[str],
) -> DataProto:
    device = torch.device("cuda")
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor(responses, device=device),
            "attention_mask": torch.tensor(attention_mask, device=device),
        },
        non_tensors={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "uid": np.array(uids, dtype=object),
        },
    )


def _make_manager(
    *,
    tokenizer: _CudaAwareTokenizer | None = None,
    teacher_client: _RecordingTeacherClient | None = None,
    rubric_client: _RecordingRubricClient | None = None,
    verifier_client: _RecordingVerifierClient | None = None,
    reward_quality_gate: dict[str, Any] | None = None,
) -> tuple[
    ROPDRewardManager,
    _CudaAwareTokenizer,
    _RecordingTeacherClient,
    _RecordingRubricClient,
    _RecordingVerifierClient,
]:
    tokenizer = tokenizer or _CudaAwareTokenizer()
    teacher_client = teacher_client or _RecordingTeacherClient()
    rubric_client = rubric_client or _RecordingRubricClient()
    verifier_client = verifier_client or _RecordingVerifierClient()
    manager = ROPDRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=None,
        pipeline=ROPDPipeline(
            teacher_client=teacher_client,
            rubric_client=rubric_client,
            verifier_client=verifier_client,
            max_pair_concurrency=1,
            max_verifier_subject_concurrency=1,
        ),
        ropd={"reward_quality_gate": reward_quality_gate or {"enabled": False}},
    )
    return manager, tokenizer, teacher_client, rubric_client, verifier_client


def test_ropd_reward_manager_runs_model_free_gpu_flow() -> None:
    data = _make_cuda_data(
        responses=[
            [101, 102, 0, 0],
            [201, 202, 203, 0],
        ],
        attention_mask=[
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        raw_prompts=["Question?", "Question?"],
        uids=["sample-0", "sample-0"],
    )
    manager, tokenizer, teacher_client, rubric_client, verifier_client = _make_manager(
        verifier_client=_RecordingVerifierClient(
            student_scores={
                ("sample-0", 0): 1.0,
                ("sample-0", 1): 1.5,
            },
        ),
    )

    output = manager(data, return_dict=True)

    reward_tensor = output["reward_tensor"]
    assert reward_tensor.device.type == "cuda"
    expected = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    expected[0, 1] = -0.5
    expected[1, 2] = -0.25
    torch.testing.assert_close(reward_tensor, expected)

    assert all(device.type == "cuda" for device in tokenizer.decoded_devices)
    assert teacher_client.calls == [("Question?", "sample-0")]
    assert [call[4] for call in rubric_client.calls] == [0, 1]
    assert [call[5] for call in verifier_client.calls] == ["teacher", "student", "teacher", "student"]

    reward_extra_info = output["reward_extra_info"]
    assert reward_extra_info["pair_index"] == [0, 1]
    assert reward_extra_info["group_size"] == [2, 2]
    assert reward_extra_info["teacher_score"] == [2.0, 2.0]
    assert reward_extra_info["student_score"] == [1.0, 1.5]
    assert reward_extra_info["reward_gap"] == [-0.5, -0.25]

    reward_control = output["reward_control"]
    assert reward_control["quality_gate_stop"] is False
    assert reward_control["update_mask"] == [True, True]


def test_ropd_compute_reward_gpu_flow_matches_trainer_boundary() -> None:
    from verl.trainer.ppo.reward import compute_reward

    data = _make_cuda_data(
        responses=[
            [101, 102, 0, 0],
            [201, 202, 203, 0],
        ],
        attention_mask=[
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        raw_prompts=["Question?", "Question?"],
        uids=["sample-0", "sample-0"],
    )
    manager, tokenizer, teacher_client, rubric_client, verifier_client = _make_manager(
        verifier_client=_RecordingVerifierClient(
            student_scores={
                ("sample-0", 0): 1.0,
                ("sample-0", 1): 1.5,
            },
        ),
    )

    reward_tensor, reward_extra_info = compute_reward(data, manager)

    assert reward_tensor.device.type == "cuda"
    expected = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    expected[0, 1] = -0.5
    expected[1, 2] = -0.25
    torch.testing.assert_close(reward_tensor, expected)
    assert reward_extra_info["pair_index"] == [0, 1]
    assert reward_extra_info["group_size"] == [2, 2]
    assert reward_extra_info["teacher_score"] == [2.0, 2.0]
    assert reward_extra_info["student_score"] == [1.0, 1.5]
    assert reward_extra_info["reward_gap"] == [-0.5, -0.25]
    assert all(device.type == "cuda" for device in tokenizer.decoded_devices)
    assert teacher_client.calls == [("Question?", "sample-0")]
    assert [call[4] for call in rubric_client.calls] == [0, 1]
    assert [call[5] for call in verifier_client.calls] == ["teacher", "student", "teacher", "student"]


def test_ropd_gpu_flow_handles_multiple_groups_and_rollouts() -> None:
    data = _make_cuda_data(
        responses=[
            [11, 12, 0, 0, 0],
            [21, 22, 23, 0, 0],
            [31, 32, 33, 34, 0],
        ],
        attention_mask=[
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        raw_prompts=["Prompt A", "Prompt A", "Prompt B"],
        uids=["group-a", "group-a", "group-b"],
    )
    manager, _, teacher_client, rubric_client, verifier_client = _make_manager(
        verifier_client=_RecordingVerifierClient(
            student_scores={
                ("group-a", 0): 1.0,
                ("group-a", 1): 1.5,
                ("group-b", 0): 0.5,
            },
        ),
    )

    output = manager(data, return_dict=True)

    expected = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    expected[0, 1] = -0.5
    expected[1, 2] = -0.25
    expected[2, 3] = -0.75
    torch.testing.assert_close(output["reward_tensor"], expected)
    assert teacher_client.calls == [("Prompt A", "group-a"), ("Prompt B", "group-b")]
    assert [(call[3], call[4]) for call in rubric_client.calls] == [
        ("group-a", 0),
        ("group-a", 1),
        ("group-b", 0),
    ]
    assert [(call[3], call[4], call[5]) for call in verifier_client.calls] == [
        ("group-a", 0, "teacher"),
        ("group-a", 0, "student"),
        ("group-a", 1, "teacher"),
        ("group-a", 1, "student"),
        ("group-b", 0, "teacher"),
        ("group-b", 0, "student"),
    ]
    reward_extra_info = output["reward_extra_info"]
    assert reward_extra_info["pair_index"] == [0, 1, 0]
    assert reward_extra_info["group_size"] == [2, 2, 1]
    assert reward_extra_info["reward_gap"] == [-0.5, -0.25, -0.75]


def test_ropd_quality_gate_retries_failed_judge_pairs() -> None:
    data = _make_cuda_data(
        responses=[
            [101, 102, 0, 0],
            [201, 202, 203, 0],
        ],
        attention_mask=[
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        raw_prompts=["Question?", "Question?"],
        uids=["sample-0", "sample-0"],
    )
    manager, _, _, _, verifier_client = _make_manager(
        verifier_client=_RecordingVerifierClient(
            student_scores={
                ("sample-0", 0): 1.0,
                ("sample-0", 1): 1.5,
            },
            fail_once={("sample-0", 1, "student")},
        ),
        reward_quality_gate={
            "enabled": True,
            "max_fallback_rate": 0.0,
            "max_retry_rounds": 1,
            "retry_pair_concurrency": 1,
            "min_group_size_after_exclusion": 1,
            "min_effective_group_rate": 1.0,
            "max_step_judge_retry_attempts": 1,
            "step_judge_retry_initial_backoff_seconds": 0.0,
            "step_judge_retry_backoff_multiplier": 1.0,
            "step_judge_retry_max_backoff_seconds": 0.0,
        },
    )

    output = manager(data, return_dict=True)

    expected = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    expected[0, 1] = -0.5
    expected[1, 2] = -0.25
    torch.testing.assert_close(output["reward_tensor"], expected)
    reward_control = output["reward_control"]
    assert reward_control["retry_round_count"] == 1
    assert reward_control["retried_pair_count"] == 1
    assert reward_control["fallback_rate_history"] == [0.5, 0.0]
    assert reward_control["quality_gate_stop"] is False
    assert reward_control["update_mask"] == [True, True]
    assert verifier_client.failed_once == {("sample-0", 1, "student")}
