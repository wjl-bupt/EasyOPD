from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import cache
from importlib import resources
from typing import Any

from easyopd.methods.ropd.pipeline import normalize_raw_prompt

PROMPT_TEMPLATE_VERSION = "phase2.v3"

_SUPPORTED_MESSAGE_ROLES = {"user", "assistant", "system", "developer"}
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


@cache
def load_prompt_template(template_name: str) -> str:
    return (
        resources.files("easyopd.methods.ropd.prompts")
        .joinpath(template_name)
        .read_text(encoding="utf-8")
    )


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if "text" in item:
                    parts.append(str(item["text"]).strip())
                    continue
                if item.get("type") == "input_text" and "text" in item:
                    parts.append(str(item["text"]).strip())
                    continue
            parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return "\n".join(part for part in parts if part)

    if isinstance(content, Mapping):
        if "text" in content:
            return str(content["text"]).strip()
        return json.dumps(dict(content), ensure_ascii=False, sort_keys=True)

    return str(content).strip()


def extract_question_text(raw_prompt: Any) -> str:
    normalized = normalize_raw_prompt(raw_prompt)
    if isinstance(normalized, str):
        return normalized.strip()

    if len(normalized) == 1:
        only_message = normalized[0]
        return _stringify_content(only_message.get("content"))

    rendered_messages: list[str] = []
    for message in normalized:
        role = str(message.get("role", "user")).strip() or "user"
        content = _stringify_content(message.get("content"))
        if not content:
            continue
        rendered_messages.append(f"{role.upper()}: {content}")

    return "\n\n".join(rendered_messages).strip()


def build_teacher_input_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    normalized = normalize_raw_prompt(raw_prompt)
    if isinstance(normalized, str):
        return [{"role": "user", "content": normalized}]

    messages: list[dict[str, Any]] = []
    for message in normalized:
        role = str(message.get("role", "user")).strip() or "user"
        if role not in _SUPPORTED_MESSAGE_ROLES:
            role = "user"

        content = message.get("content")
        if isinstance(content, str | list):
            normalized_content = content
        else:
            normalized_content = _stringify_content(content)

        messages.append({"role": role, "content": normalized_content})

    return messages


def render_template(template: str, *, replacements: Mapping[str, str]) -> str:
    unknown_placeholders = {
        match.group(1) for match in _PLACEHOLDER_PATTERN.finditer(template) if match.group(1) not in replacements
    }
    if unknown_placeholders:
        unknown_list = ", ".join(sorted(unknown_placeholders))
        raise ValueError(f"Unsupported template placeholder(s): {unknown_list}")

    return _PLACEHOLDER_PATTERN.sub(lambda match: replacements[match.group(1)], template)


__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "_stringify_content",
    "build_teacher_input_messages",
    "extract_question_text",
    "load_prompt_template",
    "render_template",
]
