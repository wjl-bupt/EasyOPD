from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from easyopd.methods.ropd.judge.config import (
    OpenAIRoleConfig,
    OpenAITransportConfig,
    ProviderCircuitBreakerConfig,
    ProviderLimitsConfig,
    RequestSchedulerConfig,
    StageBreakerConfigSet,
    build_circuit_breaker_config,
    coerce_mapping,
    coerce_non_negative_int,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_positive_int,
    coerce_optional_string,
    coerce_positive_int,
    merge_nested_mappings,
)
from easyopd.methods.ropd.judge.openai_env import apply_selected_openai_profile_to_environment
from easyopd.methods.ropd.judge.provider import SUPPORTED_ROLE_API_STYLES, SUPPORTED_ROLE_PROVIDERS
from easyopd.methods.ropd.judge.resolver import JudgeProviderResolver, ResolvedJudgeProviderConfig, ResolvedJudgeRole


@dataclass(frozen=True, slots=True)
class JudgeRoleDefaults:
    default_provider: str = "openai_compatible"
    default_model: str = "gpt-5.2-chat-latest"
    default_api_key: str | None = None
    default_base_url: str | None = None
    default_reasoning_effort: str | None = None
    default_timeout_seconds: float = 90.0
    default_max_output_tokens: int | None = 8192
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_empty_response_retries: int = 0
    default_incomplete_retries: int = 0
    default_parse_error_retries: int = 0
    default_schema_error_retries: int = 0
    default_validation_error_retries: int = 0


@dataclass(frozen=True, slots=True)
class JudgeRuntimeConfig:
    roles: dict[str, OpenAIRoleConfig]
    transport: OpenAITransportConfig
    provider_limits: ProviderLimitsConfig
    request_scheduler: RequestSchedulerConfig


@cache
def _load_repo_dotenv_into_environment() -> bool:
    skip_repo_dotenv = os.getenv("ROPD_SKIP_REPO_DOTENV")
    if skip_repo_dotenv is not None and skip_repo_dotenv.strip().lower() in {"1", "true", "yes", "on"}:
        return False
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    return load_dotenv(dotenv_path=dotenv_path, override=True)


def prepare_repo_environment() -> None:
    _load_repo_dotenv_into_environment()
    apply_selected_openai_profile_to_environment()


def _get_env_value(name: str) -> str | None:
    prepare_repo_environment()
    env_value = os.getenv(name)
    if env_value is not None and env_value.strip():
        return env_value
    return None


def resolve_profiled_provider_limits_config(provider_limits_config: Mapping[str, Any]) -> dict[str, Any]:
    resolved_provider_limits_config = {
        key: copy.deepcopy(value) for key, value in provider_limits_config.items() if key != "profiles"
    }
    selected_profile = coerce_optional_string(os.getenv("OPENAI_PROFILE"), default=None)
    if selected_profile is None:
        return resolved_provider_limits_config

    profile_overrides = coerce_mapping(provider_limits_config.get("profiles"))
    selected_profile_override = coerce_mapping(profile_overrides.get(selected_profile.upper()))
    if not selected_profile_override:
        return resolved_provider_limits_config
    return merge_nested_mappings(resolved_provider_limits_config, selected_profile_override)


def _require_api_key(api_key: str | None) -> str:
    if api_key is None or not api_key.strip():
        raise ValueError("Black-OPD OpenAI-compatible clients require a non-empty API key.")
    return api_key.strip()


def _resolve_role_env_value(
    role_config: dict[str, Any],
    *,
    value_key: str,
    env_key: str,
    default_env_name: str | None,
    default_value: str | None,
) -> str | None:
    direct_value = coerce_optional_string(role_config.get(value_key), default=None)
    if direct_value is not None:
        return direct_value

    env_name = coerce_optional_string(role_config.get(env_key), default=default_env_name)
    if env_name is not None:
        env_value = _get_env_value(env_name)
        if env_value is not None:
            return env_value
    return default_value


def _resolve_reasoning_effort(role_config: dict[str, Any], *, default: str | None) -> str | None:
    reasoning_config = coerce_mapping(role_config.get("reasoning"))
    reasoning_value = role_config.get("reasoning_effort", reasoning_config.get("effort"))
    return coerce_optional_string(reasoning_value, default=default)


def build_role_config(
    role_config: dict[str, Any],
    *,
    role_name: Literal["teacher", "rubricator", "verifier"],
    default_provider: str,
    default_model: str,
    default_api_key: str | None,
    default_base_url: str | None,
    default_reasoning_effort: str | None,
    default_timeout_seconds: float,
    default_max_output_tokens: int | None,
    default_temperature: float | None,
    default_top_p: float | None,
    default_empty_response_retries: int,
    default_incomplete_retries: int,
    default_parse_error_retries: int,
    default_schema_error_retries: int,
    default_validation_error_retries: int,
) -> OpenAIRoleConfig:
    provider = coerce_optional_string(role_config.get("provider"), default=default_provider) or default_provider
    if provider not in SUPPORTED_ROLE_PROVIDERS:
        raise ValueError(
            f"Unsupported Black-OPD provider {provider!r}. Supported providers: {sorted(SUPPORTED_ROLE_PROVIDERS)}."
        )
    api_style = coerce_optional_string(role_config.get("api_style"), default="responses") or "responses"
    if api_style not in SUPPORTED_ROLE_API_STYLES:
        raise ValueError(
            f"Unsupported Black-OPD api_style {api_style!r}. Supported values: {sorted(SUPPORTED_ROLE_API_STYLES)}."
        )

    model = coerce_optional_string(role_config.get("model"), default=default_model) or default_model
    reasoning_effort = _resolve_reasoning_effort(role_config, default=default_reasoning_effort)
    timeout_seconds = coerce_optional_float(role_config.get("timeout_seconds"), default=default_timeout_seconds)
    max_output_tokens = coerce_optional_int(role_config.get("max_output_tokens"), default=default_max_output_tokens)
    temperature = coerce_optional_float(role_config.get("temperature"), default=default_temperature)
    top_p = coerce_optional_float(role_config.get("top_p"), default=default_top_p)
    response_retry_config = coerce_mapping(role_config.get("response_retry"))
    empty_response_retries = coerce_non_negative_int(
        role_config.get("empty_response_retries", response_retry_config.get("empty_response_retries")),
        default=default_empty_response_retries,
        field_name=f"ropd.{role_name}.response_retry.empty_response_retries",
    )
    incomplete_retries = coerce_non_negative_int(
        role_config.get("incomplete_retries", response_retry_config.get("incomplete_retries")),
        default=default_incomplete_retries,
        field_name=f"ropd.{role_name}.response_retry.incomplete_retries",
    )
    parse_error_retries = coerce_non_negative_int(
        role_config.get("parse_error_retries", response_retry_config.get("parse_error_retries")),
        default=default_parse_error_retries,
        field_name=f"ropd.{role_name}.response_retry.parse_error_retries",
    )
    schema_error_retries = coerce_non_negative_int(
        role_config.get("schema_error_retries", response_retry_config.get("schema_error_retries")),
        default=default_schema_error_retries,
        field_name=f"ropd.{role_name}.response_retry.schema_error_retries",
    )
    validation_error_retries = coerce_non_negative_int(
        role_config.get("validation_error_retries", response_retry_config.get("validation_error_retries")),
        default=default_validation_error_retries,
        field_name=f"ropd.{role_name}.response_retry.validation_error_retries",
    )
    index_path = coerce_optional_string(role_config.get("index_path"), default=None)

    api_key = _resolve_role_env_value(
        role_config,
        value_key="api_key",
        env_key="api_key_env",
        default_env_name="OPENAI_API_KEY",
        default_value=default_api_key,
    )
    base_url = _resolve_role_env_value(
        role_config,
        value_key="base_url",
        env_key="base_url_env",
        default_env_name="OPENAI_BASE_URL",
        default_value=default_base_url,
    )

    if provider == "offline_index":
        if role_name != "teacher":
            raise ValueError("offline_index is only supported for teacher.")
        if index_path is None:
            raise ValueError("ropd.teacher.index_path is required when teacher.provider=offline_index.")

    if provider == "openai_compatible":
        resolved_api_key = _require_api_key(api_key)
    else:
        resolved_api_key = api_key or ""

    return OpenAIRoleConfig(
        provider=provider,
        api_style=api_style,
        model=model,
        api_key=resolved_api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds or default_timeout_seconds,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        empty_response_retries=empty_response_retries,
        incomplete_retries=incomplete_retries,
        parse_error_retries=parse_error_retries,
        schema_error_retries=schema_error_retries,
        validation_error_retries=validation_error_retries,
        index_path=index_path,
    )


def role_config_from_resolved_role(role: ResolvedJudgeRole) -> OpenAIRoleConfig:
    return OpenAIRoleConfig(
        provider=role.provider,
        api_style=role.api_style,
        model=role.model,
        api_key=role.api_key or "",
        base_url=role.base_url,
        timeout_seconds=role.timeout_seconds,
        reasoning_effort=role.reasoning_effort,
        max_output_tokens=role.max_output_tokens,
        temperature=role.temperature,
        top_p=role.top_p,
        empty_response_retries=role.response_retry.get("empty_response_retries", 0),
        incomplete_retries=role.response_retry.get("incomplete_retries", 0),
        parse_error_retries=role.response_retry.get("parse_error_retries", 0),
        schema_error_retries=role.response_retry.get("schema_error_retries", 0),
        validation_error_retries=role.response_retry.get("validation_error_retries", 0),
        index_path=role.index_path,
    )


def build_judge_runtime_config(
    *,
    config: Mapping[str, Any] | None,
    role_defaults: Mapping[str, JudgeRoleDefaults],
    field_name_prefix: str,
    resolver_spec_path_default: str,
    resolver_entrypoint_default: str = "train",
    inherit_credentials_from: str | None = "teacher",
) -> JudgeRuntimeConfig:
    if not role_defaults:
        raise ValueError("role_defaults must specify at least one role.")
    prepare_repo_environment()
    config = {} if config is None else dict(config)
    resolution_config = coerce_mapping(config.get("provider_resolution"))
    role_overrides = {role_name: coerce_mapping(config.get(role_name)) for role_name in role_defaults}
    transport_config = coerce_mapping(config.get("transport"))
    provider_limits_config = resolve_profiled_provider_limits_config(coerce_mapping(config.get("provider_limits")))
    request_scheduler_config = coerce_mapping(config.get("request_scheduler"))

    roles: dict[str, OpenAIRoleConfig] = {}
    if resolution_config:
        merged_resolution_overrides = merge_nested_mappings(
            coerce_mapping(resolution_config.get("overrides")),
            {role_name: role_overrides[role_name] for role_name in role_defaults if role_overrides[role_name]},
        )
        resolved_provider_config = JudgeProviderResolver(
            spec_path=(
                coerce_optional_string(resolution_config.get("spec_path"), default=resolver_spec_path_default)
                or resolver_spec_path_default
            ),
            entrypoint=(
                coerce_optional_string(resolution_config.get("entrypoint"), default=resolver_entrypoint_default)
                or resolver_entrypoint_default
            ),
            overrides=merged_resolution_overrides,
        ).resolve()
        for role_name in role_defaults:
            roles[role_name] = role_config_from_resolved_role(getattr(resolved_provider_config.roles, role_name))
        primary_online_role = _select_primary_online_resolved_role_for_roles(resolved_provider_config, role_defaults)
        transport_config = merge_nested_mappings(primary_online_role.transport, transport_config)
        provider_limits_config = resolve_profiled_provider_limits_config(
            merge_nested_mappings(primary_online_role.provider_limits, provider_limits_config)
        )
    else:
        if not any(role_overrides.values()):
            role_field_names = " or ".join(f"{field_name_prefix}.{role_name}" for role_name in role_defaults)
            raise ValueError(f"{field_name_prefix}.provider_resolution or {role_field_names} is required.")
        primary_role_name = inherit_credentials_from if inherit_credentials_from in role_defaults else next(iter(role_defaults))
        primary_defaults = role_defaults[primary_role_name]
        primary_role = build_role_config(
            role_overrides[primary_role_name],
            role_name=primary_role_name,  # type: ignore[arg-type]
            default_provider=primary_defaults.default_provider,
            default_model=primary_defaults.default_model,
            default_api_key=primary_defaults.default_api_key,
            default_base_url=primary_defaults.default_base_url,
            default_reasoning_effort=primary_defaults.default_reasoning_effort,
            default_timeout_seconds=primary_defaults.default_timeout_seconds,
            default_max_output_tokens=primary_defaults.default_max_output_tokens,
            default_temperature=primary_defaults.default_temperature,
            default_top_p=primary_defaults.default_top_p,
            default_empty_response_retries=primary_defaults.default_empty_response_retries,
            default_incomplete_retries=primary_defaults.default_incomplete_retries,
            default_parse_error_retries=primary_defaults.default_parse_error_retries,
            default_schema_error_retries=primary_defaults.default_schema_error_retries,
            default_validation_error_retries=primary_defaults.default_validation_error_retries,
        )
        roles[primary_role_name] = primary_role
        for role_name, defaults in role_defaults.items():
            if role_name == primary_role_name:
                continue
            fallback_provider = "openai_compatible" if primary_role.provider == "offline_index" else primary_role.provider
            role_override = role_overrides[role_name]
            roles[role_name] = build_role_config(
                role_override,
                role_name=role_name,  # type: ignore[arg-type]
                default_provider=(
                    fallback_provider if defaults.default_provider == "openai_compatible" else defaults.default_provider
                ),
                default_model=role_override.get("model", primary_role.model),
                default_api_key=defaults.default_api_key if defaults.default_api_key is not None else primary_role.api_key,
                default_base_url=defaults.default_base_url if defaults.default_base_url is not None else primary_role.base_url,
                default_reasoning_effort=defaults.default_reasoning_effort,
                default_timeout_seconds=defaults.default_timeout_seconds,
                default_max_output_tokens=defaults.default_max_output_tokens,
                default_temperature=defaults.default_temperature,
                default_top_p=defaults.default_top_p,
                default_empty_response_retries=defaults.default_empty_response_retries,
                default_incomplete_retries=defaults.default_incomplete_retries,
                default_parse_error_retries=defaults.default_parse_error_retries,
                default_schema_error_retries=defaults.default_schema_error_retries,
                default_validation_error_retries=defaults.default_validation_error_retries,
            )

    transport = OpenAITransportConfig(
        max_retries=int(transport_config.get("max_retries", 2)),
        initial_backoff_seconds=float(transport_config.get("initial_backoff_seconds", 1.0)),
        backoff_multiplier=float(transport_config.get("backoff_multiplier", 2.0)),
        max_backoff_seconds=float(transport_config.get("max_backoff_seconds", 8.0)),
        max_in_flight_requests=coerce_positive_int(
            transport_config.get("max_in_flight_requests"),
            default=32,
            field_name=f"{field_name_prefix}.transport.max_in_flight_requests",
        ),
    )
    circuit_breaker = build_circuit_breaker_config(
        provider_limits_config.get("circuit_breaker"),
        defaults=ProviderCircuitBreakerConfig(),
        field_name_prefix=f"{field_name_prefix}.provider_limits.circuit_breaker",
    )
    stage_breaker_config = coerce_mapping(provider_limits_config.get("stage_breakers"))
    provider_limits = ProviderLimitsConfig(
        max_concurrent_requests=coerce_positive_int(
            provider_limits_config.get("max_concurrent_requests"),
            default=transport.max_in_flight_requests,
            field_name=f"{field_name_prefix}.provider_limits.max_concurrent_requests",
        ),
        max_rpm=coerce_optional_positive_int(
            provider_limits_config.get("max_rpm"),
            default=1920,
            field_name=f"{field_name_prefix}.provider_limits.max_rpm",
        ),
        max_tpm=coerce_optional_positive_int(
            provider_limits_config.get("max_tpm"),
            default=6000000,
            field_name=f"{field_name_prefix}.provider_limits.max_tpm",
        ),
        circuit_breaker=circuit_breaker,
        stage_breakers=StageBreakerConfigSet(
            teacher=build_circuit_breaker_config(
                stage_breaker_config.get("teacher"),
                defaults=circuit_breaker,
                field_name_prefix=f"{field_name_prefix}.provider_limits.stage_breakers.teacher",
            ),
            rubricator=build_circuit_breaker_config(
                stage_breaker_config.get("rubricator"),
                defaults=circuit_breaker,
                field_name_prefix=f"{field_name_prefix}.provider_limits.stage_breakers.rubricator",
            ),
            verifier=build_circuit_breaker_config(
                stage_breaker_config.get("verifier"),
                defaults=circuit_breaker,
                field_name_prefix=f"{field_name_prefix}.provider_limits.stage_breakers.verifier",
            ),
        ),
    )
    request_scheduler = RequestSchedulerConfig(
        enabled=bool(request_scheduler_config.get("enabled", True)),
        num_workers=coerce_optional_positive_int(
            request_scheduler_config.get("num_workers"),
            default=None,
            field_name=f"{field_name_prefix}.request_scheduler.num_workers",
        ),
        max_queue_size=coerce_optional_positive_int(
            request_scheduler_config.get("max_queue_size"),
            default=None,
            field_name=f"{field_name_prefix}.request_scheduler.max_queue_size",
        ),
        stage_priority_enabled=bool(request_scheduler_config.get("stage_priority_enabled", True)),
        record_queue_metrics=bool(request_scheduler_config.get("record_queue_metrics", True)),
    )
    return JudgeRuntimeConfig(
        roles=roles,
        transport=transport,
        provider_limits=provider_limits,
        request_scheduler=request_scheduler,
    )


def _select_primary_online_resolved_role_for_roles(
    resolved_provider_config: ResolvedJudgeProviderConfig,
    role_names: Mapping[str, JudgeRoleDefaults],
) -> ResolvedJudgeRole:
    fallback_role_name = next(iter(role_names))
    fallback_role = getattr(resolved_provider_config.roles, fallback_role_name)
    for role_name in role_names:
        role = getattr(resolved_provider_config.roles, role_name)
        if role.provider == "openai_compatible":
            return role
    return fallback_role


__all__ = [
    "JudgeRoleDefaults",
    "JudgeRuntimeConfig",
    "build_judge_runtime_config",
    "build_role_config",
    "load_dotenv",
    "prepare_repo_environment",
    "resolve_profiled_provider_limits_config",
    "role_config_from_resolved_role",
    "_get_env_value",
    "_load_repo_dotenv_into_environment",
    "_require_api_key",
    "_resolve_reasoning_effort",
    "_resolve_role_env_value",
]
