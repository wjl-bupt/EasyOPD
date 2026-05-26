from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

RUBRIC_SCHEMA_VERSION = "ropd.rubric.v1"
VERIFIER_SCHEMA_VERSION = "ropd.verifier.v1"
MIN_RUBRIC_CRITERIA = 4
MAX_RUBRIC_CRITERIA = 12


class JudgeClientError(RuntimeError):
    def __init__(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        error_type: Literal[
            "timeout",
            "http_error",
            "circuit_open",
            "parse_error",
            "schema_error",
            "empty_response",
            "incomplete",
            "validation_error",
        ],
        message: str,
        status_code: int | None = None,
        retriable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.error_type = error_type
        self.status_code = status_code
        self.retriable = retriable
        self.details = dict(details or {})

    def add_context(self, **context: Any) -> JudgeClientError:
        for key, value in context.items():
            if value is not None:
                self.details[key] = value
        return self

    def clone(self) -> JudgeClientError:
        return JudgeClientError(
            stage=self.stage,
            error_type=self.error_type,
            message=str(self),
            status_code=self.status_code,
            retriable=self.retriable,
            details=copy.deepcopy(self.details),
        )


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    category: str
    criterion: str
    points: int

    @field_validator("criterion_id", "category", "criterion")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("criterion_id")
    @classmethod
    def _validate_criterion_id(cls, value: str) -> str:
        if not value.startswith("c") or not value[1:].isdigit():
            raise ValueError("criterion_id must use the form c1, c2, c3, ...")
        return value

    @field_validator("points")
    @classmethod
    def _validate_points(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("points must be between 1 and 5")
        return value


class StructuredRubric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[RUBRIC_SCHEMA_VERSION]
    rubrics: list[RubricCriterion]
    maximum_score: int

    def canonical_hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rubrics": [criterion.model_dump(mode="json") for criterion in self.rubrics],
            "maximum_score": self.maximum_score,
        }

    @property
    def rubric_hash(self) -> str:
        canonical_json = json.dumps(
            self.canonical_hash_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @property
    def total_points(self) -> int:
        return sum(criterion.points for criterion in self.rubrics)


class VerifierScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[VERIFIER_SCHEMA_VERSION]
    judgement: list[bool]
    final_score: float


def _json_schema_for_model(model: type[BaseModel], *, name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": name,
        "schema": model.model_json_schema(),
        "strict": True,
    }


def _strip_markdown_json_fence(raw_text: str) -> str:
    stripped_text = raw_text.strip()
    if not stripped_text.startswith("```"):
        return raw_text

    first_newline_index = stripped_text.find("\n")
    if first_newline_index < 0:
        return raw_text

    fence_language = stripped_text[3:first_newline_index].strip().lower()
    if fence_language not in {"", "json"}:
        return raw_text

    fenced_body = stripped_text[first_newline_index + 1 :]
    if not fenced_body.endswith("```"):
        return raw_text
    return fenced_body[:-3].strip()


def _parse_json_payload(
    raw_text: str,
    *,
    stage: Literal["rubricator", "verifier"],
) -> dict[str, Any]:
    normalized_raw_text = _strip_markdown_json_fence(raw_text)
    try:
        payload = json.loads(normalized_raw_text)
    except json.JSONDecodeError as exc:
        raise JudgeClientError(
            stage=stage,
            error_type="parse_error",
            message=f"{stage} returned invalid JSON: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise JudgeClientError(
            stage=stage,
            error_type="schema_error",
            message=f"{stage} returned a non-object JSON payload.",
        )
    return payload


def _validate_structured_rubric(rubric: StructuredRubric) -> StructuredRubric:
    if len(rubric.rubrics) < MIN_RUBRIC_CRITERIA or len(rubric.rubrics) > MAX_RUBRIC_CRITERIA:
        raise JudgeClientError(
            stage="rubricator",
            error_type="validation_error",
            message=(
                "rubricator returned an invalid number of rubric criteria "
                f"(expected {MIN_RUBRIC_CRITERIA} to {MAX_RUBRIC_CRITERIA})."
            ),
        )
    criterion_ids = [criterion.criterion_id for criterion in rubric.rubrics]
    if len(set(criterion_ids)) != len(criterion_ids):
        raise JudgeClientError(
            stage="rubricator",
            error_type="validation_error",
            message="rubricator returned duplicate criterion_id values.",
        )
    if rubric.maximum_score != rubric.total_points:
        rubric = rubric.model_copy(update={"maximum_score": rubric.total_points})
    return rubric


def _normalize_structured_rubric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized_payload = copy.deepcopy(payload)
    rubrics = normalized_payload.get("rubrics")
    if isinstance(rubrics, list):
        for criterion in rubrics:
            if not isinstance(criterion, dict):
                continue
            criterion_id = criterion.get("criterion_id")
            if isinstance(criterion_id, str):
                stripped_id = criterion_id.strip()
                if len(stripped_id) > 1 and stripped_id[0].lower() == "c" and stripped_id[1:].isdigit():
                    criterion["criterion_id"] = stripped_id.lower()
    return normalized_payload


def _parse_structured_rubric(raw_text: str) -> StructuredRubric:
    payload = _normalize_structured_rubric_payload(_parse_json_payload(raw_text, stage="rubricator"))
    try:
        rubric = StructuredRubric.model_validate(payload)
    except ValidationError as exc:
        raise JudgeClientError(
            stage="rubricator",
            error_type="schema_error",
            message=f"rubricator payload does not match the rubric schema: {exc}",
        ) from exc
    return _validate_structured_rubric(rubric)


def _validate_verifier_score(
    score: VerifierScore,
    *,
    rubric: StructuredRubric,
) -> VerifierScore:
    if len(score.judgement) != len(rubric.rubrics):
        raise JudgeClientError(
            stage="verifier",
            error_type="validation_error",
            message="verifier judgement length does not match rubric length.",
        )
    recomputed_final_score = sum(
        criterion.points for criterion, judgement in zip(rubric.rubrics, score.judgement, strict=True) if judgement
    )
    if float(score.final_score) != float(recomputed_final_score):
        score = score.model_copy(update={"final_score": float(recomputed_final_score)})
    return score


def _parse_verifier_score(raw_text: str, *, rubric: StructuredRubric) -> VerifierScore:
    payload = _parse_json_payload(raw_text, stage="verifier")
    try:
        score = VerifierScore.model_validate(payload)
    except ValidationError as exc:
        raise JudgeClientError(
            stage="verifier",
            error_type="schema_error",
            message=f"verifier payload does not match the verifier schema: {exc}",
        ) from exc
    return _validate_verifier_score(score, rubric=rubric)


__all__ = [
    "MAX_RUBRIC_CRITERIA",
    "MIN_RUBRIC_CRITERIA",
    "RUBRIC_SCHEMA_VERSION",
    "VERIFIER_SCHEMA_VERSION",
    "JudgeClientError",
    "RubricCriterion",
    "StructuredRubric",
    "VerifierScore",
    "_json_schema_for_model",
    "_normalize_structured_rubric_payload",
    "_parse_json_payload",
    "_parse_structured_rubric",
    "_parse_verifier_score",
    "_strip_markdown_json_fence",
    "_validate_structured_rubric",
    "_validate_verifier_score",
]
