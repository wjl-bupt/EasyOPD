"""ROPD verifier replay utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from easyopd.methods.ropd.judge.schema import JudgeClientError, StructuredRubric, VerifierScore

ROPDClientError = JudgeClientError
ROPDStructuredRubric = StructuredRubric
ROPDVerifierScore = VerifierScore

VerifierSubject = Literal["teacher", "student"]


@dataclass(frozen=True, slots=True)
class VerifierReplaySample:
    uid: str
    pair_index: int
    raw_prompt: Any
    teacher_answer: str
    student_answer: str
    rubric_hash: str
    rubric_model: str
    rubric: ROPDStructuredRubric

    def answer_for_subject(self, subject: VerifierSubject) -> str:
        return self.teacher_answer if subject == "teacher" else self.student_answer


class ReplayVerifierClient(Protocol):
    def score(
        self,
        raw_prompt: Any,
        rubric: ROPDStructuredRubric,
        answer: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
        subject: str | None = None,
    ) -> ROPDVerifierScore: ...


def load_verifier_replay_samples(
    pair_traces_path: str | Path,
    rubrics_dir: str | Path,
    *,
    limit: int | None = None,
    uid: str | None = None,
    pair_indices: set[int] | None = None,
) -> tuple[VerifierReplaySample, ...]:
    resolved_pair_traces_path = Path(pair_traces_path)
    resolved_rubrics_dir = Path(rubrics_dir)
    samples: list[VerifierReplaySample] = []

    for line in resolved_pair_traces_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("error_stage") != "verifier":
            continue
        if uid is not None and str(entry.get("uid")) != uid:
            continue
        pair_index = int(entry["pair_index"])
        if pair_indices is not None and pair_index not in pair_indices:
            continue
        rubric_hash = str(entry["rubric_hash"])
        rubric_path = resolved_rubrics_dir / f"{rubric_hash}.json"
        rubric = ROPDStructuredRubric.model_validate(json.loads(rubric_path.read_text(encoding="utf-8")))
        samples.append(
            VerifierReplaySample(
                uid=str(entry["uid"]),
                pair_index=pair_index,
                raw_prompt=entry["raw_prompt"],
                teacher_answer=str(entry["teacher_answer"]),
                student_answer=str(entry["student_answer"]),
                rubric_hash=rubric_hash,
                rubric_model=str(entry.get("rubric_model", "")),
                rubric=rubric,
            )
        )
        if limit is not None and len(samples) >= limit:
            break

    return tuple(samples)


def replay_verifier_samples(
    samples: tuple[VerifierReplaySample, ...],
    verifier_client: ReplayVerifierClient,
    *,
    attempts: int,
    subjects: tuple[VerifierSubject, ...],
) -> list[dict[str, Any]]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")

    replay_results: list[dict[str, Any]] = []
    for sample in samples:
        for attempt_index in range(attempts):
            for subject in subjects:
                replay_results.append(
                    replay_verifier_once(
                        sample,
                        verifier_client,
                        attempt_index=attempt_index,
                        subject=subject,
                    )
                )
    return replay_results


def replay_verifier_once(
    sample: VerifierReplaySample,
    verifier_client: ReplayVerifierClient,
    *,
    attempt_index: int,
    subject: VerifierSubject,
) -> dict[str, Any]:
    answer = sample.answer_for_subject(subject)
    base_record = {
        "uid": sample.uid,
        "pair_index": sample.pair_index,
        "rubric_hash": sample.rubric_hash,
        "rubric_model": sample.rubric_model,
        "subject": subject,
        "attempt_index": attempt_index,
    }
    try:
        score = verifier_client.score(
            sample.raw_prompt,
            sample.rubric,
            answer,
            uid=sample.uid,
            pair_index=sample.pair_index,
            subject=subject,
        )
    except ROPDClientError as exc:
        return {
            **base_record,
            "ok": False,
            "error_stage": exc.stage,
            "error_type": exc.error_type,
            "message": str(exc),
            "error_details": dict(exc.details),
        }

    return {
        **base_record,
        "ok": True,
        "final_score": float(score.final_score),
        "judgement": list(score.judgement),
    }


def summarize_replay_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_attempts": len(results),
        "successes": sum(bool(record.get("ok")) for record in results),
        "failures": sum(not bool(record.get("ok")) for record in results),
        "by_subject": {},
        "failure_types": {},
    }
    for subject in ("teacher", "student"):
        subject_records = [record for record in results if record.get("subject") == subject]
        if not subject_records:
            continue
        summary["by_subject"][subject] = {
            "total_attempts": len(subject_records),
            "successes": sum(bool(record.get("ok")) for record in subject_records),
            "failures": sum(not bool(record.get("ok")) for record in subject_records),
        }

    for record in results:
        if record.get("ok"):
            continue
        error_type = str(record.get("error_type", "unknown"))
        summary["failure_types"][error_type] = summary["failure_types"].get(error_type, 0) + 1

    return summary


__all__ = [
    "ReplayVerifierClient",
    "VerifierReplaySample",
    "VerifierSubject",
    "load_verifier_replay_samples",
    "replay_verifier_once",
    "replay_verifier_samples",
    "summarize_replay_results",
]
