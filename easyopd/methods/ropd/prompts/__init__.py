"""ROPD prompt builders.

Generic prompt rendering and raw-prompt text extraction live in
`easyopd.methods.ropd.prompt_utils`; this module builds the rubricator and
verifier prompts used by the ROPD reward path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from easyopd.methods.ropd.pipeline import normalize_raw_prompt as _normalize_raw_prompt
from easyopd.methods.ropd.prompt_utils import (
    PROMPT_TEMPLATE_VERSION,
    _stringify_content,
    build_teacher_input_messages,
    extract_question_text,
    load_prompt_template,
    render_template,
)

_load_template = load_prompt_template
_render_template = render_template


def _dump_prompt_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _looks_like_skywork_model(model: str | None) -> bool:
    if model is None:
        return False
    return "skywork" in model.strip().lower()


def _resolve_verifier_template_name(model: str | None) -> str:
    if _looks_like_skywork_model(model):
        return "verifier_skywork.txt"
    return "verifier.txt"


def _count_rubrics(rubrics: Any) -> int | None:
    if isinstance(rubrics, list | tuple):
        return len(rubrics)
    return None


def _render_ordered_rubrics(rubrics: Any) -> str:
    if not isinstance(rubrics, list | tuple):
        return _dump_prompt_json(rubrics)

    rendered_items: list[str] = []
    for index, rubric in enumerate(rubrics, start=1):
        if isinstance(rubric, Mapping):
            criterion_id = str(rubric.get("criterion_id", f"c{index}")).strip() or f"c{index}"
            points = rubric.get("points", "")
            category = str(rubric.get("category", "")).strip()
            criterion = str(rubric.get("criterion", "")).strip()
            rendered_items.append(
                f"{index}. [{criterion_id}] points={points} | category={category} | criterion={criterion}"
            )
            continue
        rendered_items.append(f"{index}. {_dump_prompt_json(rubric)}")
    return "\n".join(rendered_items)


def _render_judgement_slot_mapping(rubrics: Any) -> str:
    if not isinstance(rubrics, list | tuple):
        return "judgement[0] -> unknown"

    rendered_items: list[str] = []
    for index, rubric in enumerate(rubrics):
        if isinstance(rubric, Mapping):
            criterion_id = str(rubric.get("criterion_id", f"c{index + 1}")).strip() or f"c{index + 1}"
        else:
            criterion_id = f"c{index + 1}"
        rendered_items.append(f"judgement[{index}] -> {criterion_id}")
    return "\n".join(rendered_items)


def build_rubricator_prompt(raw_prompt: Any, *, teacher_response: str, student_response: str) -> str:
    template = _load_template("rubricator.txt")
    return _render_template(
        template,
        replacements={
            "question": extract_question_text(raw_prompt),
            "teacher_response": teacher_response,
            "student_response": student_response,
        },
    )


def build_verifier_prompt(raw_prompt: Any, *, response: str, rubrics: Any, model: str | None = None) -> str:
    template = _load_template(_resolve_verifier_template_name(model))
    rubric_count = _count_rubrics(rubrics)
    return _render_template(
        template,
        replacements={
            "question": extract_question_text(raw_prompt),
            "resp": response,
            "rubrics": _dump_prompt_json(rubrics),
            "ordered_rubrics": _render_ordered_rubrics(rubrics),
            "judgement_slot_mapping": _render_judgement_slot_mapping(rubrics),
            "rubric_count": "unknown" if rubric_count is None else str(rubric_count),
        },
    )


__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "_normalize_raw_prompt",
    "_render_template",
    "_stringify_content",
    "build_rubricator_prompt",
    "build_teacher_input_messages",
    "build_verifier_prompt",
    "extract_question_text",
]
