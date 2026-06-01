"""Judge runtime facade.

This module re-exports canonical judge runtime symbols from focused submodules.
It also hosts `build_judge_runtime_config()`, `JudgeRoleDefaults`, and
`JudgeRuntimeConfig`, which are the preferred APIs for assembling judge role
configs.

New code should import provider/runtime primitives from
`algo.judge.provider`, `algo.judge.teacher_client`, `algo.judge.rate_limit`,
and `algo.judge.circuit_breaker`. Imports from this facade remain supported
for compatibility.
"""

from __future__ import annotations

from typing import Any

from easyopd.methods.ropd.judge import runtime_builder as _runtime_builder
from easyopd.methods.ropd.judge.circuit_breaker import ProviderCircuitBreaker
from easyopd.methods.ropd.judge.provider import (
    PROVIDER_METRIC_NAMES,
    REQUEST_FINGERPRINT_SCHEMA_VERSION,
    SUPPORTED_ROLE_API_STYLES,
    SUPPORTED_ROLE_PROVIDERS,
    TRANSIENT_HTTP_STATUS_CODES,
    OpenAICompatibleProvider,
    SharedProviderResources,
    StageProviderRuntime,
    _empty_provider_metrics,
    _sum_provider_metrics,
)
from easyopd.methods.ropd.judge.rate_limit import SyncTokenBucket
from easyopd.methods.ropd.judge.runtime_builder import JudgeRoleDefaults, JudgeRuntimeConfig
from easyopd.methods.ropd.judge.teacher_client import OpenAITeacherClient, StaticTeacherClient

load_dotenv = _runtime_builder.load_dotenv


def _sync_runtime_builder_load_dotenv() -> None:
    _runtime_builder.load_dotenv = load_dotenv


def _load_repo_dotenv_into_environment() -> bool:
    _sync_runtime_builder_load_dotenv()
    return _runtime_builder._load_repo_dotenv_into_environment()


_load_repo_dotenv_into_environment.cache_clear = _runtime_builder._load_repo_dotenv_into_environment.cache_clear  # type: ignore[attr-defined]


def prepare_repo_environment() -> None:
    _sync_runtime_builder_load_dotenv()
    _runtime_builder.prepare_repo_environment()


def _get_env_value(name: str) -> str | None:
    _sync_runtime_builder_load_dotenv()
    return _runtime_builder._get_env_value(name)


def build_role_config(*args: Any, **kwargs: Any) -> Any:
    _sync_runtime_builder_load_dotenv()
    return _runtime_builder.build_role_config(*args, **kwargs)


def build_judge_runtime_config(*args: Any, **kwargs: Any) -> JudgeRuntimeConfig:
    _sync_runtime_builder_load_dotenv()
    return _runtime_builder.build_judge_runtime_config(*args, **kwargs)


def _resolve_role_env_value(*args: Any, **kwargs: Any) -> str | None:
    _sync_runtime_builder_load_dotenv()
    return _runtime_builder._resolve_role_env_value(*args, **kwargs)


resolve_profiled_provider_limits_config = _runtime_builder.resolve_profiled_provider_limits_config
role_config_from_resolved_role = _runtime_builder.role_config_from_resolved_role
_require_api_key = _runtime_builder._require_api_key

__all__ = [
    "PROVIDER_METRIC_NAMES",
    "REQUEST_FINGERPRINT_SCHEMA_VERSION",
    "SUPPORTED_ROLE_API_STYLES",
    "SUPPORTED_ROLE_PROVIDERS",
    "TRANSIENT_HTTP_STATUS_CODES",
    "JudgeRoleDefaults",
    "JudgeRuntimeConfig",
    "OpenAICompatibleProvider",
    "OpenAITeacherClient",
    "ProviderCircuitBreaker",
    "SharedProviderResources",
    "StageProviderRuntime",
    "StaticTeacherClient",
    "SyncTokenBucket",
    "build_judge_runtime_config",
    "build_role_config",
    "load_dotenv",
    "prepare_repo_environment",
    "resolve_profiled_provider_limits_config",
    "role_config_from_resolved_role",
    "_empty_provider_metrics",
    "_get_env_value",
    "_load_repo_dotenv_into_environment",
    "_require_api_key",
    "_resolve_role_env_value",
    "_sum_provider_metrics",
]
