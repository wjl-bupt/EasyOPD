from __future__ import annotations

from typing import Any

from easyopd.methods.ropd.judge.config import JudgeDebugConfig, OpenAIRoleConfig
from easyopd.methods.ropd.judge.provider import OpenAICompatibleProvider
from easyopd.methods.ropd.judge.schema import JudgeClientError
from easyopd.methods.ropd.prompt_utils import build_teacher_input_messages


class OpenAITeacherClient:
    def __init__(self, *, provider: OpenAICompatibleProvider, role_config: OpenAIRoleConfig) -> None:
        self.provider = provider
        self.role_config = role_config

    def generate(self, raw_prompt: Any, *, uid: str | None = None) -> str:
        try:
            input_messages = build_teacher_input_messages(raw_prompt)
        except (TypeError, ValueError) as exc:
            raise JudgeClientError(
                stage="teacher",
                error_type="validation_error",
                message=f"teacher prompt construction failed: {exc}",
            ) from exc
        try:
            return self.provider.create_text(stage="teacher", role=self.role_config, input_payload=input_messages)
        except JudgeClientError as exc:
            exc.add_context(uid=uid)
            raise


class StaticTeacherClient:
    def __init__(self, *, debug_config: JudgeDebugConfig, role_config: OpenAIRoleConfig) -> None:
        self.debug_config = debug_config
        self.role_config = role_config

    def generate(self, raw_prompt: Any, *, uid: str | None = None) -> str:
        del raw_prompt, uid
        return self.debug_config.static_teacher_response


__all__ = ["OpenAITeacherClient", "StaticTeacherClient"]
