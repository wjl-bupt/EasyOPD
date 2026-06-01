from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from easyopd.methods.ropd.judge.scheduler import RequestSchedulerConfig

__all__ = (
    "JUDGE_STAGES",
    "TEXT_ARTIFACT_MODES",
    "JudgeDebugConfig",
    "OpenAIRoleConfig",
    "OpenAITransportConfig",
    "ProviderCircuitBreakerConfig",
    "ProviderLimitsConfig",
    "RequestSchedulerConfig",
    "StageBreakerConfigSet",
    "build_circuit_breaker_config",
    "coerce_bool",
    "coerce_mapping",
    "coerce_non_negative_int",
    "coerce_optional_float",
    "coerce_optional_int",
    "coerce_optional_positive_int",
    "coerce_optional_string",
    "coerce_positive_int",
    "merge_nested_mappings",
)

JUDGE_STAGES: tuple[Literal["teacher", "rubricator", "verifier"], ...] = ("teacher", "rubricator", "verifier")
TEXT_ARTIFACT_MODES: tuple[Literal["diagnostic_only", "all_pairs"], ...] = ("diagnostic_only", "all_pairs")


@dataclass(frozen=True, slots=True)
class OpenAIRoleConfig:
    model: str
    api_key: str
    base_url: str | None
    timeout_seconds: float
    provider: str = "openai_compatible"
    api_style: str = "responses"
    reasoning_effort: str | None = "none"
    max_output_tokens: int | None = None
    temperature: float | None = 1.0
    top_p: float | None = None
    empty_response_retries: int = 0
    incomplete_retries: int = 0
    parse_error_retries: int = 0
    schema_error_retries: int = 0
    validation_error_retries: int = 0
    index_path: str | None = None


@dataclass(frozen=True, slots=True)
class OpenAITransportConfig:
    max_retries: int = 2
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0
    max_in_flight_requests: int = 32


@dataclass(frozen=True, slots=True)
class ProviderCircuitBreakerConfig:
    consecutive_retriable_errors: int = 5
    rolling_window_size: int = 20
    rolling_error_rate: float = 0.2
    cooldown_seconds: float = 30.0
    half_open_probe_requests: int = 2


@dataclass(frozen=True, slots=True)
class StageBreakerConfigSet:
    teacher: ProviderCircuitBreakerConfig
    rubricator: ProviderCircuitBreakerConfig
    verifier: ProviderCircuitBreakerConfig

    def for_stage(
        self,
        stage: Literal["teacher", "rubricator", "verifier"],
    ) -> ProviderCircuitBreakerConfig:
        return getattr(self, stage)


@dataclass(frozen=True, slots=True)
class ProviderLimitsConfig:
    max_concurrent_requests: int = 32
    max_rpm: int | None = 1920
    max_tpm: int | None = 6000000
    circuit_breaker: ProviderCircuitBreakerConfig = field(default_factory=ProviderCircuitBreakerConfig)
    stage_breakers: StageBreakerConfigSet | None = None

    def __post_init__(self) -> None:
        if self.stage_breakers is None:
            shared_breakers = StageBreakerConfigSet(
                teacher=self.circuit_breaker,
                rubricator=self.circuit_breaker,
                verifier=self.circuit_breaker,
            )
            object.__setattr__(self, "stage_breakers", shared_breakers)


@dataclass(frozen=True, slots=True)
class JudgeDebugConfig:
    include_text_artifacts: bool = False
    output_dir: str = "outputs/ropd"
    retention_days: int = 14
    text_artifact_mode: Literal["diagnostic_only", "all_pairs"] = "diagnostic_only"
    static_teacher_response: str = "Teacher reference answer: 42."
    static_maximum_score: int = 8


def coerce_optional_float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return float(stripped)
    return float(value)


def coerce_optional_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return int(stripped)
    return int(value)


def coerce_optional_string(value: Any, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    stripped = str(value).strip()
    return stripped or default


def coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    stripped = str(value).strip().lower()
    if stripped in {"1", "true", "yes", "on"}:
        return True
    if stripped in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot coerce {value!r} to bool.")


def coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def merge_nested_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: copy.deepcopy(value) for key, value in base.items()}
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = merge_nested_mappings(base_value, override_value)
        else:
            merged[key] = copy.deepcopy(override_value)
    return merged


def coerce_positive_int(value: Any, *, default: int, field_name: str) -> int:
    resolved_value = int(value if value is not None else default)
    if resolved_value < 1:
        raise ValueError(f"{field_name} must be at least 1.")
    return resolved_value


def coerce_non_negative_int(value: Any, *, default: int, field_name: str) -> int:
    resolved_value = int(value if value is not None else default)
    if resolved_value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return resolved_value


def coerce_optional_positive_int(value: Any, *, default: int | None, field_name: str) -> int | None:
    resolved_value = coerce_optional_int(value, default=default)
    if resolved_value is None:
        return None
    if resolved_value < 1:
        raise ValueError(f"{field_name} must be at least 1.")
    return resolved_value


def build_circuit_breaker_config(
    breaker_config: Mapping[str, Any] | None,
    *,
    defaults: ProviderCircuitBreakerConfig,
    field_name_prefix: str,
) -> ProviderCircuitBreakerConfig:
    resolved_breaker_config = coerce_mapping(breaker_config)
    rolling_error_rate = float(resolved_breaker_config.get("rolling_error_rate", defaults.rolling_error_rate))
    if not 0.0 < rolling_error_rate <= 1.0:
        raise ValueError(f"{field_name_prefix}.rolling_error_rate must be in (0, 1].")
    built_config = ProviderCircuitBreakerConfig(
        consecutive_retriable_errors=coerce_positive_int(
            resolved_breaker_config.get("consecutive_retriable_errors"),
            default=defaults.consecutive_retriable_errors,
            field_name=f"{field_name_prefix}.consecutive_retriable_errors",
        ),
        rolling_window_size=coerce_positive_int(
            resolved_breaker_config.get("rolling_window_size"),
            default=defaults.rolling_window_size,
            field_name=f"{field_name_prefix}.rolling_window_size",
        ),
        rolling_error_rate=rolling_error_rate,
        cooldown_seconds=float(resolved_breaker_config.get("cooldown_seconds", defaults.cooldown_seconds)),
        half_open_probe_requests=coerce_positive_int(
            resolved_breaker_config.get("half_open_probe_requests"),
            default=defaults.half_open_probe_requests,
            field_name=f"{field_name_prefix}.half_open_probe_requests",
        ),
    )
    if built_config.cooldown_seconds < 0:
        raise ValueError(f"{field_name_prefix}.cooldown_seconds must be non-negative.")
    return built_config
