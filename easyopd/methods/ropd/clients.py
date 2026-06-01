"""ROPD judge clients and config builders.

Shared judge config/schema/provider infrastructure lives in
`easyopd.methods.ropd.judge.*`. This module wires those building blocks into
the teacher/rubricator/verifier client triple that ROPD's reward manager
consumes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import easyopd.methods.ropd.judge.runtime as _judge_clients
from easyopd.methods.ropd.artifacts import ROPDArtifactExporter, ROPDExportConfig
from easyopd.methods.ropd.prompts import (
    PROMPT_TEMPLATE_VERSION,
    build_rubricator_prompt,
    build_verifier_prompt,
)
from easyopd.methods.ropd.judge.config import (
    TEXT_ARTIFACT_MODES,
    JudgeDebugConfig,
    OpenAIRoleConfig,
    OpenAITransportConfig,
    ProviderCircuitBreakerConfig,
    ProviderLimitsConfig,
    RequestSchedulerConfig,
    StageBreakerConfigSet,
    build_circuit_breaker_config,
    coerce_bool,
    coerce_mapping,
    coerce_optional_positive_int,
    coerce_optional_string,
    coerce_positive_int,
    merge_nested_mappings,
)
from easyopd.methods.ropd.judge.resolver import JudgeProviderResolver, ResolvedJudgeProviderConfig, ResolvedJudgeRole
from easyopd.methods.ropd.judge.runtime import (
    TRANSIENT_HTTP_STATUS_CODES as TRANSIENT_HTTP_STATUS_CODES,
)
from easyopd.methods.ropd.judge.runtime import (
    OpenAICompatibleProvider,
    OpenAITeacherClient,
    StaticTeacherClient,
)
from easyopd.methods.ropd.judge.runtime import (
    SyncTokenBucket as SyncTokenBucket,
)
from easyopd.methods.ropd.judge.schema import (
    MAX_RUBRIC_CRITERIA,
    MIN_RUBRIC_CRITERIA,
    RUBRIC_SCHEMA_VERSION,
    VERIFIER_SCHEMA_VERSION,
    JudgeClientError,
    RubricCriterion,
    StructuredRubric,
    VerifierScore,
    _json_schema_for_model,
    _normalize_structured_rubric_payload,
    _parse_json_payload,
    _parse_structured_rubric,
    _parse_verifier_score,
    _strip_markdown_json_fence,
    _validate_structured_rubric,
    _validate_verifier_score,
)
from easyopd.methods.ropd.teacher_index import OfflineTeacherIndex, OfflineTeacherIndexClient, build_teacher_fingerprint_payload

load_dotenv = _judge_clients.load_dotenv

ROPDClientError = JudgeClientError
ROPDDebugConfig = JudgeDebugConfig
ROPDProviderCircuitBreakerConfig = ProviderCircuitBreakerConfig
ROPDProviderLimitsConfig = ProviderLimitsConfig
ROPDRequestSchedulerConfig = RequestSchedulerConfig
ROPDStageBreakerConfigSet = StageBreakerConfigSet
ROPDRubricCriterion = RubricCriterion
ROPDStructuredRubric = StructuredRubric
ROPDVerifierScore = VerifierScore


def _sync_legacy_load_dotenv_patch() -> None:
    _judge_clients.load_dotenv = load_dotenv


def _load_repo_dotenv_into_environment() -> bool:
    _sync_legacy_load_dotenv_patch()
    return _judge_clients._load_repo_dotenv_into_environment()


_load_repo_dotenv_into_environment.cache_clear = _judge_clients._load_repo_dotenv_into_environment.cache_clear  # type: ignore[attr-defined]


def _prepare_repo_environment() -> None:
    _sync_legacy_load_dotenv_patch()
    _judge_clients.prepare_repo_environment()


def _build_role_config(*args: Any, **kwargs: Any) -> OpenAIRoleConfig:
    _sync_legacy_load_dotenv_patch()
    return _judge_clients.build_role_config(*args, **kwargs)


_resolve_profiled_provider_limits_config = _judge_clients.resolve_profiled_provider_limits_config
_role_config_from_resolved_role = _judge_clients.role_config_from_resolved_role
def _select_primary_online_resolved_role(resolved_config: ResolvedJudgeProviderConfig) -> ResolvedJudgeRole:
    for role in (resolved_config.roles.teacher, resolved_config.roles.rubricator, resolved_config.roles.verifier):
        if role.provider == "openai_compatible":
            return role
    return resolved_config.roles.teacher


@dataclass(frozen=True, slots=True)
class ROPDClientConfig:
    teacher: OpenAIRoleConfig
    rubricator: OpenAIRoleConfig
    verifier: OpenAIRoleConfig
    transport: OpenAITransportConfig = field(default_factory=OpenAITransportConfig)
    max_group_concurrency: int = 4
    max_pair_concurrency: int = 8
    max_verifier_subject_concurrency: int = 2
    provider_limits: ROPDProviderLimitsConfig = field(default_factory=ROPDProviderLimitsConfig)
    request_scheduler: ROPDRequestSchedulerConfig = field(default_factory=ROPDRequestSchedulerConfig)
    export: ROPDExportConfig = field(default_factory=ROPDExportConfig)
    debug: ROPDDebugConfig = field(default_factory=ROPDDebugConfig)




def _build_resolution_overrides_from_legacy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for role_name in ("teacher", "rubricator", "verifier"):
        role_config = coerce_mapping(config.get(role_name))
        if role_config:
            overrides[role_name] = role_config
    return overrides


def build_ropd_client_config(config: dict[str, Any] | None = None) -> ROPDClientConfig:
    _prepare_repo_environment()
    config = {} if config is None else dict(config)
    resolution_config = coerce_mapping(config.get("provider_resolution"))
    transport_config = coerce_mapping(config.get("transport"))
    provider_limits_config = _resolve_profiled_provider_limits_config(coerce_mapping(config.get("provider_limits")))
    request_scheduler_config = coerce_mapping(config.get("request_scheduler"))
    export_config = coerce_mapping(config.get("export"))
    debug_config = coerce_mapping(config.get("debug"))

    if not resolution_config and not config:
        raise ValueError("ropd.provider_resolution is required.")

    if resolution_config:
        merged_resolution_overrides = merge_nested_mappings(
            coerce_mapping(resolution_config.get("overrides")),
            _build_resolution_overrides_from_legacy_config(config),
        )
        resolved_provider_config = JudgeProviderResolver(
            spec_path=(
                coerce_optional_string(
                    resolution_config.get("spec_path"),
                    default="easyopd/config/ropd/judge_providers.yaml",
                )
                or "easyopd/config/ropd/judge_providers.yaml"
            ),
            entrypoint=coerce_optional_string(resolution_config.get("entrypoint"), default="train") or "train",
            overrides=merged_resolution_overrides,
        ).resolve()
        teacher_role = _role_config_from_resolved_role(resolved_provider_config.roles.teacher)
        rubricator_role = _role_config_from_resolved_role(resolved_provider_config.roles.rubricator)
        verifier_role = _role_config_from_resolved_role(resolved_provider_config.roles.verifier)
        primary_online_role = _select_primary_online_resolved_role(resolved_provider_config)
        transport_config = merge_nested_mappings(primary_online_role.transport, transport_config)
        provider_limits_config = _resolve_profiled_provider_limits_config(
            merge_nested_mappings(primary_online_role.provider_limits, provider_limits_config)
        )
    else:
        teacher_config = coerce_mapping(config.get("teacher"))
        rubricator_config = coerce_mapping(config.get("rubricator"))
        verifier_config = coerce_mapping(config.get("verifier"))
        teacher_role = _build_role_config(
            teacher_config,
            role_name="teacher",
            default_provider="openai_compatible",
            default_model="gpt-5.2-chat-latest",
            default_api_key=None,
            default_base_url=None,
            default_reasoning_effort=None,
            default_timeout_seconds=45.0,
            default_max_output_tokens=8192,
            default_temperature=None,
            default_top_p=None,
            default_empty_response_retries=0,
            default_incomplete_retries=0,
            default_parse_error_retries=0,
            default_schema_error_retries=0,
            default_validation_error_retries=0,
        )
        rubricator_role = _build_role_config(
            rubricator_config,
            role_name="rubricator",
            default_provider="openai_compatible" if teacher_role.provider == "offline_index" else teacher_role.provider,
            default_model=teacher_role.model,
            default_api_key=teacher_role.api_key,
            default_base_url=teacher_role.base_url,
            default_reasoning_effort=None,
            default_timeout_seconds=90.0,
            default_max_output_tokens=8192,
            default_temperature=None,
            default_top_p=None,
            default_empty_response_retries=0,
            default_incomplete_retries=0,
            default_parse_error_retries=0,
            default_schema_error_retries=0,
            default_validation_error_retries=2,
        )
        verifier_role = _build_role_config(
            verifier_config,
            role_name="verifier",
            default_provider="openai_compatible" if teacher_role.provider == "offline_index" else teacher_role.provider,
            default_model=teacher_role.model,
            default_api_key=teacher_role.api_key,
            default_base_url=teacher_role.base_url,
            default_reasoning_effort=None,
            default_timeout_seconds=30.0,
            default_max_output_tokens=8192,
            default_temperature=None,
            default_top_p=None,
            default_empty_response_retries=1,
            default_incomplete_retries=1,
            default_parse_error_retries=1,
            default_schema_error_retries=1,
            default_validation_error_retries=0,
        )

    transport = OpenAITransportConfig(
        max_retries=int(transport_config.get("max_retries", 2)),
        initial_backoff_seconds=float(transport_config.get("initial_backoff_seconds", 1.0)),
        backoff_multiplier=float(transport_config.get("backoff_multiplier", 2.0)),
        max_backoff_seconds=float(transport_config.get("max_backoff_seconds", 8.0)),
        max_in_flight_requests=coerce_positive_int(
            transport_config.get("max_in_flight_requests"),
            default=32,
            field_name="ropd.transport.max_in_flight_requests",
        ),
    )
    if transport.max_retries < 0:
        raise ValueError("ropd.transport.max_retries must be non-negative.")
    if transport.initial_backoff_seconds < 0:
        raise ValueError("ropd.transport.initial_backoff_seconds must be non-negative.")
    if transport.backoff_multiplier < 1.0:
        raise ValueError("ropd.transport.backoff_multiplier must be at least 1.0.")
    if transport.max_backoff_seconds < 0:
        raise ValueError("ropd.transport.max_backoff_seconds must be non-negative.")

    max_group_concurrency = coerce_positive_int(
        config.get("max_group_concurrency"),
        default=4,
        field_name="ropd.max_group_concurrency",
    )
    max_pair_concurrency = coerce_positive_int(
        config.get("max_pair_concurrency"),
        default=8,
        field_name="ropd.max_pair_concurrency",
    )
    max_verifier_subject_concurrency = coerce_positive_int(
        config.get("max_verifier_subject_concurrency"),
        default=2,
        field_name="ropd.max_verifier_subject_concurrency",
    )

    circuit_breaker = build_circuit_breaker_config(
        provider_limits_config.get("circuit_breaker"),
        defaults=ROPDProviderCircuitBreakerConfig(),
        field_name_prefix="ropd.provider_limits.circuit_breaker",
    )
    stage_breaker_config = coerce_mapping(provider_limits_config.get("stage_breakers"))
    provider_limits = ROPDProviderLimitsConfig(
        max_concurrent_requests=coerce_positive_int(
            provider_limits_config.get("max_concurrent_requests"),
            default=transport.max_in_flight_requests,
            field_name="ropd.provider_limits.max_concurrent_requests",
        ),
        max_rpm=coerce_optional_positive_int(
            provider_limits_config.get("max_rpm"),
            default=1920,
            field_name="ropd.provider_limits.max_rpm",
        ),
        max_tpm=coerce_optional_positive_int(
            provider_limits_config.get("max_tpm"),
            default=6000000,
            field_name="ropd.provider_limits.max_tpm",
        ),
        circuit_breaker=circuit_breaker,
        stage_breakers=ROPDStageBreakerConfigSet(
            teacher=build_circuit_breaker_config(
                stage_breaker_config.get("teacher"),
                defaults=circuit_breaker,
                field_name_prefix="ropd.provider_limits.stage_breakers.teacher",
            ),
            rubricator=build_circuit_breaker_config(
                stage_breaker_config.get("rubricator"),
                defaults=circuit_breaker,
                field_name_prefix="ropd.provider_limits.stage_breakers.rubricator",
            ),
            verifier=build_circuit_breaker_config(
                stage_breaker_config.get("verifier"),
                defaults=circuit_breaker,
                field_name_prefix="ropd.provider_limits.stage_breakers.verifier",
            ),
        ),
    )
    request_scheduler = ROPDRequestSchedulerConfig(
        enabled=coerce_bool(request_scheduler_config.get("enabled"), default=True),
        num_workers=coerce_optional_positive_int(
            request_scheduler_config.get("num_workers"),
            default=None,
            field_name="ropd.request_scheduler.num_workers",
        ),
        max_queue_size=coerce_optional_positive_int(
            request_scheduler_config.get("max_queue_size"),
            default=None,
            field_name="ropd.request_scheduler.max_queue_size",
        ),
        stage_priority_enabled=coerce_bool(
            request_scheduler_config.get("stage_priority_enabled"),
            default=True,
        ),
        record_queue_metrics=coerce_bool(
            request_scheduler_config.get("record_queue_metrics"),
            default=True,
        ),
    )

    include_text_artifacts = coerce_bool(
        debug_config.get("include_text_artifacts"),
        default=coerce_bool(export_config.get("enabled"), default=False),
    )
    output_dir = coerce_optional_string(
        debug_config.get("output_dir"),
        default=coerce_optional_string(export_config.get("output_dir"), default="outputs/ropd"),
    ) or "outputs/ropd"
    static_maximum_score = int(debug_config.get("static_maximum_score", 8))
    if static_maximum_score < 6:
        raise ValueError("ropd.debug.static_maximum_score must be at least 6.")
    retention_days = int(debug_config.get("retention_days", export_config.get("retention_days", 14)))
    if retention_days < 1:
        raise ValueError("ropd.debug.retention_days must be at least 1.")
    text_artifact_mode = (
        coerce_optional_string(
            debug_config.get("text_artifact_mode"),
            default=coerce_optional_string(
                export_config.get("text_artifact_mode"),
                default="diagnostic_only",
            ),
        )
        or "diagnostic_only"
    )
    if text_artifact_mode not in TEXT_ARTIFACT_MODES:
        raise ValueError(
            "ropd.debug.text_artifact_mode must be one of "
            f"{list(TEXT_ARTIFACT_MODES)!r}, got {text_artifact_mode!r}."
        )
    debug = ROPDDebugConfig(
        include_text_artifacts=include_text_artifacts,
        output_dir=output_dir,
        retention_days=retention_days,
        text_artifact_mode=text_artifact_mode,
        static_teacher_response=(
            coerce_optional_string(
                debug_config.get("static_teacher_response"),
                default="Teacher reference answer: 42.",
            )
            or "Teacher reference answer: 42."
        ),
        static_maximum_score=static_maximum_score,
    )
    return ROPDClientConfig(
        teacher=teacher_role,
        rubricator=rubricator_role,
        verifier=verifier_role,
        transport=transport,
        max_group_concurrency=max_group_concurrency,
        max_pair_concurrency=max_pair_concurrency,
        max_verifier_subject_concurrency=max_verifier_subject_concurrency,
        provider_limits=provider_limits,
        request_scheduler=request_scheduler,
        export=ROPDExportConfig(
            enabled=include_text_artifacts,
            output_dir=output_dir,
            retention_days=retention_days,
            text_artifact_mode=text_artifact_mode,
        ),
        debug=debug,
    )




class OpenAIRubricatorClient:
    def __init__(self, *, provider: OpenAICompatibleProvider, role_config: OpenAIRoleConfig) -> None:
        self.provider = provider
        self.role_config = role_config

    def generate(
        self,
        raw_prompt: Any,
        teacher_response: str,
        student_response: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
    ) -> ROPDStructuredRubric:
        try:
            prompt_text = build_rubricator_prompt(
                raw_prompt,
                teacher_response=teacher_response,
                student_response=student_response,
            )
        except (TypeError, ValueError) as exc:
            raise ROPDClientError(
                stage="rubricator",
                error_type="validation_error",
                message=f"rubricator prompt construction failed: {exc}",
            ) from exc

        try:
            return self.provider.create_text(
                stage="rubricator",
                role=self.role_config,
                input_payload=prompt_text,
                text_format=_json_schema_for_model(ROPDStructuredRubric, name="ropd_rubric"),
                output_validator=_parse_structured_rubric,
            )
        except ROPDClientError as exc:
            exc.add_context(uid=uid, pair_index=pair_index)
            raise


class OpenAIVerifierClient:
    def __init__(self, *, provider: OpenAICompatibleProvider, role_config: OpenAIRoleConfig) -> None:
        self.provider = provider
        self.role_config = role_config

    def score(
        self,
        raw_prompt: Any,
        rubric: ROPDStructuredRubric,
        answer: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
        subject: str | None = None,
    ) -> ROPDVerifierScore:
        try:
            prompt_text = build_verifier_prompt(
                raw_prompt,
                response=answer,
                rubrics=[criterion.model_dump(mode="json") for criterion in rubric.rubrics],
                model=self.role_config.model,
            )
        except (TypeError, ValueError) as exc:
            raise ROPDClientError(
                stage="verifier",
                error_type="validation_error",
                message=f"verifier prompt construction failed: {exc}",
            ) from exc

        try:
            return self.provider.create_text(
                stage="verifier",
                role=self.role_config,
                input_payload=prompt_text,
                text_format=_json_schema_for_model(ROPDVerifierScore, name="ropd_verifier"),
                output_validator=lambda raw_text: _parse_verifier_score(raw_text, rubric=rubric),
            )
        except ROPDClientError as exc:
            exc.add_context(uid=uid, pair_index=pair_index, subject=subject)
            raise


def _build_static_rubric_criteria(maximum_score: int) -> list[ROPDRubricCriterion]:
    return [
        ROPDRubricCriterion(
            criterion_id=f"c{index}",
            category="Correctness",
            criterion=f"criterion {index}",
            points=1,
        )
        for index in range(1, maximum_score + 1)
    ]


def _resolve_static_student_score(maximum_score: int, pair_index: int | None) -> int:
    max_student_score = max(1, maximum_score // 2 - 1)
    base_score = 2 + max(pair_index or 0, 0)
    return min(base_score, max_student_score)




class StaticRubricatorClient:
    def __init__(self, *, debug_config: ROPDDebugConfig, role_config: OpenAIRoleConfig) -> None:
        self.debug_config = debug_config
        self.role_config = role_config
        self._criteria = _build_static_rubric_criteria(self.debug_config.static_maximum_score)

    def generate(
        self,
        raw_prompt: Any,
        teacher_response: str,
        student_response: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
    ) -> ROPDStructuredRubric:
        del raw_prompt, teacher_response, student_response, uid
        rubric = ROPDStructuredRubric(
            schema_version=RUBRIC_SCHEMA_VERSION,
            rubrics=self._criteria,
            maximum_score=self.debug_config.static_maximum_score,
        )
        return _validate_structured_rubric(rubric)


class StaticVerifierClient:
    def __init__(self, *, debug_config: ROPDDebugConfig, role_config: OpenAIRoleConfig) -> None:
        self.debug_config = debug_config
        self.role_config = role_config

    def score(
        self,
        raw_prompt: Any,
        rubric: ROPDStructuredRubric,
        answer: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
        subject: str | None = None,
    ) -> ROPDVerifierScore:
        del raw_prompt, answer, uid
        student_score = _resolve_static_student_score(rubric.maximum_score, pair_index)
        teacher_score = 1 if student_score > 0 else 0
        resolved_score = teacher_score if subject == "teacher" else student_score
        judgement = [True] * int(resolved_score) + [False] * (len(rubric.rubrics) - int(resolved_score))
        score = ROPDVerifierScore(
            schema_version=VERIFIER_SCHEMA_VERSION,
            judgement=judgement,
            final_score=float(resolved_score),
        )
        return _validate_verifier_score(score, rubric=rubric)


def _build_role_client(
    *,
    stage: Literal["teacher", "rubricator", "verifier"],
    role_config: OpenAIRoleConfig,
    debug_config: ROPDDebugConfig,
    provider: OpenAICompatibleProvider | None,
) -> Any:
    if role_config.provider == "offline_index":
        if stage != "teacher":
            raise ValueError("offline_index is only supported for teacher.")
        fingerprint = build_teacher_fingerprint_payload(
            provider="openai_compatible",
            model=role_config.model,
            base_url=role_config.base_url,
            reasoning_effort=role_config.reasoning_effort,
            max_output_tokens=role_config.max_output_tokens,
            temperature=role_config.temperature,
            top_p=role_config.top_p,
            timeout_seconds=role_config.timeout_seconds,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        teacher_index = OfflineTeacherIndex.load(
            index_path=role_config.index_path,
            expected_fingerprint=fingerprint,
        )
        return OfflineTeacherIndexClient(teacher_index=teacher_index)

    if role_config.provider == "openai_compatible":
        if provider is None:
            raise ValueError("OpenAI-compatible Black-OPD roles require an initialized provider.")
        if stage == "teacher":
            return OpenAITeacherClient(provider=provider, role_config=role_config)
        if stage == "rubricator":
            return OpenAIRubricatorClient(provider=provider, role_config=role_config)
        return OpenAIVerifierClient(provider=provider, role_config=role_config)

    if role_config.provider == "static":
        if stage == "teacher":
            return StaticTeacherClient(debug_config=debug_config, role_config=role_config)
        if stage == "rubricator":
            return StaticRubricatorClient(debug_config=debug_config, role_config=role_config)
        return StaticVerifierClient(debug_config=debug_config, role_config=role_config)

    raise ValueError(f"Unsupported Black-OPD role provider {role_config.provider!r}.")


def build_ropd_pipeline(
    config: ROPDClientConfig | dict[str, Any] | None = None,
    *,
    provider: OpenAICompatibleProvider | None = None,
) -> Any:
    from easyopd.methods.ropd.pipeline import ROPDPipeline

    resolved_config = config if isinstance(config, ROPDClientConfig) else build_ropd_client_config(config)
    requires_openai_provider = any(
        role.provider == "openai_compatible"
        for role in (resolved_config.teacher, resolved_config.rubricator, resolved_config.verifier)
    )
    resolved_provider = (
        provider
        or (
            OpenAICompatibleProvider(
                resolved_config.transport,
                provider_limits=resolved_config.provider_limits,
                request_scheduler_config=resolved_config.request_scheduler,
            )
            if requires_openai_provider
            else None
        )
    )
    return ROPDPipeline(
        teacher_client=_build_role_client(
            stage="teacher",
            role_config=resolved_config.teacher,
            debug_config=resolved_config.debug,
            provider=resolved_provider,
        ),
        rubric_client=_build_role_client(
            stage="rubricator",
            role_config=resolved_config.rubricator,
            debug_config=resolved_config.debug,
            provider=resolved_provider,
        ),
        verifier_client=_build_role_client(
            stage="verifier",
            role_config=resolved_config.verifier,
            debug_config=resolved_config.debug,
            provider=resolved_provider,
        ),
        max_pair_concurrency=resolved_config.max_pair_concurrency,
        max_verifier_subject_concurrency=resolved_config.max_verifier_subject_concurrency,
        artifact_exporter=(
            ROPDArtifactExporter(resolved_config.export) if resolved_config.export.enabled else None
        ),
    )


__all__ = [
    "MAX_RUBRIC_CRITERIA",
    "MIN_RUBRIC_CRITERIA",
    "OpenAICompatibleProvider",
    "OpenAIRoleConfig",
    "OpenAIRubricatorClient",
    "OpenAITeacherClient",
    "OpenAITransportConfig",
    "OpenAIVerifierClient",
    "PROMPT_TEMPLATE_VERSION",
    "ROPDClientConfig",
    "ROPDClientError",
    "ROPDDebugConfig",
    "ROPDExportConfig",
    "ROPDProviderCircuitBreakerConfig",
    "ROPDProviderLimitsConfig",
    "ROPDRequestSchedulerConfig",
    "ROPDRubricCriterion",
    "ROPDStageBreakerConfigSet",
    "ROPDStructuredRubric",
    "ROPDVerifierScore",
    "RUBRIC_SCHEMA_VERSION",
    "TRANSIENT_HTTP_STATUS_CODES",
    "VERIFIER_SCHEMA_VERSION",
    "_json_schema_for_model",
    "_normalize_structured_rubric_payload",
    "_parse_json_payload",
    "_parse_structured_rubric",
    "_parse_verifier_score",
    "_strip_markdown_json_fence",
    "_validate_structured_rubric",
    "_validate_verifier_score",
    "build_ropd_client_config",
    "build_ropd_pipeline",
]
