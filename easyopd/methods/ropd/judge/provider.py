from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Mapping
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock
from typing import Any, Literal

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from easyopd.methods.ropd.judge.circuit_breaker import ProviderCircuitBreaker
from easyopd.methods.ropd.judge.config import (
    JUDGE_STAGES,
    OpenAIRoleConfig,
    OpenAITransportConfig,
    ProviderCircuitBreakerConfig,
    ProviderLimitsConfig,
    RequestSchedulerConfig,
    StageBreakerConfigSet,
    coerce_optional_float,
    coerce_optional_string,
)
from easyopd.methods.ropd.judge.rate_limit import SyncTokenBucket
from easyopd.methods.ropd.judge.scheduler import BoundedRequestScheduler
from easyopd.methods.ropd.judge.schema import JudgeClientError
from easyopd.methods.ropd.prompt_utils import PROMPT_TEMPLATE_VERSION

REQUEST_FINGERPRINT_SCHEMA_VERSION = "ropd.request_fingerprint.v1"
TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
SUPPORTED_ROLE_PROVIDERS = frozenset({"openai_compatible", "static", "offline_index"})
SUPPORTED_ROLE_API_STYLES = frozenset({"responses", "chat_completions"})
PROVIDER_METRIC_NAMES = (
    "requests_started",
    "requests_succeeded",
    "requests_failed",
    "retries",
    "retriable_errors",
    "timeout_errors",
    "rate_limit_wait_count",
    "rate_limit_wait_seconds",
    "circuit_open_rejections",
    "estimated_tokens",
)


@dataclass(slots=True)
class SharedProviderResources:
    semaphore: BoundedSemaphore
    rpm_limiter: SyncTokenBucket | None
    tpm_limiter: SyncTokenBucket | None
    client_cache: dict[tuple[str, str | None, float], Any]
    client_cache_lock: Lock
    inflight_requests: dict[str, Future[Any]]
    inflight_requests_lock: Lock


@dataclass(slots=True)
class StageProviderRuntime:
    stage_name: Literal["teacher", "rubricator", "verifier"]
    circuit_breaker: ProviderCircuitBreaker
    metrics: dict[str, float]
    breaker_config: ProviderCircuitBreakerConfig
    first_retriable_error: dict[str, Any] | None = None
    last_retriable_error: dict[str, Any] | None = None


def _empty_provider_metrics() -> dict[str, float]:
    return {name: 0.0 for name in PROVIDER_METRIC_NAMES}


def _sum_provider_metrics(metric_snapshots: list[dict[str, float]]) -> dict[str, float]:
    totals = _empty_provider_metrics()
    for snapshot in metric_snapshots:
        for name in PROVIDER_METRIC_NAMES:
            totals[name] += snapshot.get(name, 0.0)
    return totals


class OpenAICompatibleProvider:
    def __init__(
        self,
        transport: OpenAITransportConfig,
        *,
        provider_limits: ProviderLimitsConfig | None = None,
        request_scheduler_config: RequestSchedulerConfig | None = None,
        client_factory: Any = OpenAI,
        sleep: Any = time.sleep,
        time_fn: Any = time.monotonic,
        uniform: Any = random.uniform,
    ) -> None:
        self.transport = transport
        self.provider_limits = provider_limits or ProviderLimitsConfig()
        self.request_scheduler_config = request_scheduler_config or RequestSchedulerConfig()
        self._client_factory = client_factory
        self._sleep = sleep
        self._time_fn = time_fn
        self._uniform = uniform
        self._metrics_lock = Lock()
        effective_max_in_flight = min(
            self.transport.max_in_flight_requests,
            self.provider_limits.max_concurrent_requests,
        )
        self._request_scheduler = self._build_request_scheduler(effective_max_in_flight)
        self._shared_resources = SharedProviderResources(
            semaphore=BoundedSemaphore(value=effective_max_in_flight),
            rpm_limiter=(
                SyncTokenBucket(
                    rate_limit=self.provider_limits.max_rpm / 60.0,
                    max_tokens=self.provider_limits.max_rpm / 60.0,
                    time_fn=self._time_fn,
                    sleep=self._sleep,
                )
                if self.provider_limits.max_rpm is not None
                else None
            ),
            tpm_limiter=(
                SyncTokenBucket(
                    rate_limit=self.provider_limits.max_tpm / 60.0,
                    max_tokens=self.provider_limits.max_tpm / 60.0,
                    time_fn=self._time_fn,
                    sleep=self._sleep,
                )
                if self.provider_limits.max_tpm is not None
                else None
            ),
            client_cache={},
            client_cache_lock=Lock(),
            inflight_requests={},
            inflight_requests_lock=Lock(),
        )
        stage_breakers = self.provider_limits.stage_breakers or StageBreakerConfigSet(
            teacher=self.provider_limits.circuit_breaker,
            rubricator=self.provider_limits.circuit_breaker,
            verifier=self.provider_limits.circuit_breaker,
        )
        self._stage_runtimes = {
            stage_name: StageProviderRuntime(
                stage_name=stage_name,
                circuit_breaker=ProviderCircuitBreaker(
                    stage_breakers.for_stage(stage_name),
                    time_fn=self._time_fn,
                ),
                metrics=_empty_provider_metrics(),
                breaker_config=stage_breakers.for_stage(stage_name),
            )
            for stage_name in JUDGE_STAGES
        }

    def _build_request_scheduler(self, effective_max_in_flight: int) -> BoundedRequestScheduler | None:
        if not self.request_scheduler_config.enabled:
            return None
        resolved_num_workers = self.request_scheduler_config.num_workers or effective_max_in_flight
        resolved_max_queue_size = self.request_scheduler_config.max_queue_size or resolved_num_workers
        return BoundedRequestScheduler(
            num_workers=resolved_num_workers,
            max_queue_size=resolved_max_queue_size,
            stage_priority_enabled=self.request_scheduler_config.stage_priority_enabled,
            record_queue_metrics=self.request_scheduler_config.record_queue_metrics,
            time_fn=self._time_fn,
        )

    def close(self) -> None:
        if self._request_scheduler is not None:
            self._request_scheduler.shutdown(wait=True)

    def snapshot_metrics(self) -> dict[str, dict[str, Any]]:
        with self._metrics_lock:
            stage_snapshots: dict[str, dict[str, Any]] = {}
            for stage_name, runtime in self._stage_runtimes.items():
                stage_snapshot: dict[str, Any] = dict(runtime.metrics)
                if runtime.first_retriable_error is not None:
                    stage_snapshot["first_retriable_error"] = copy.deepcopy(runtime.first_retriable_error)
                if runtime.last_retriable_error is not None:
                    stage_snapshot["last_retriable_error"] = copy.deepcopy(runtime.last_retriable_error)
                stage_snapshots[stage_name] = stage_snapshot
        stage_snapshots["totals"] = _sum_provider_metrics(list(stage_snapshots.values()))
        if self._request_scheduler is not None and self.request_scheduler_config.record_queue_metrics:
            stage_snapshots["totals"].update(self._request_scheduler.snapshot_metrics())
        return stage_snapshots

    def _stage_runtime(
        self,
        stage: Literal["teacher", "rubricator", "verifier"],
    ) -> StageProviderRuntime:
        return self._stage_runtimes[stage]

    def create_text(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None = None,
        output_validator: Any = None,
    ) -> Any:
        request_fingerprint = self._build_request_fingerprint(
            stage=stage,
            role=role,
            input_payload=input_payload,
            text_format=text_format,
        )

        def run_direct_request() -> Any:
            return self._create_text_direct(
                stage=stage,
                role=role,
                input_payload=input_payload,
                text_format=text_format,
                output_validator=output_validator,
            )

        if request_fingerprint is not None:
            future, is_leader = self._acquire_inflight_request(request_fingerprint)
            if not is_leader:
                return self._await_inflight_result(future)
        else:
            future = None

        if self._request_scheduler is None:
            try:
                result = run_direct_request()
            except BaseException as exc:
                if request_fingerprint is not None and future is not None:
                    self._complete_inflight_request(request_fingerprint=request_fingerprint, future=future, error=exc)
                raise
            if request_fingerprint is not None and future is not None:
                self._complete_inflight_request(
                    request_fingerprint=request_fingerprint,
                    future=future,
                    result=result,
                )
            return result

        try:
            result = self._request_scheduler.submit(stage=stage, fn=run_direct_request)
        except BaseException as exc:
            if request_fingerprint is not None and future is not None:
                self._complete_inflight_request(request_fingerprint=request_fingerprint, future=future, error=exc)
            raise
        if request_fingerprint is not None and future is not None:
            self._complete_inflight_request(
                request_fingerprint=request_fingerprint,
                future=future,
                result=result,
            )
        return result

    def _create_text_direct(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
        output_validator: Any = None,
    ) -> Any:
        request_kwargs = self._build_request_kwargs(
            role=role,
            input_payload=input_payload,
            text_format=text_format,
        )
        request_metadata = self._build_request_metadata(role=role, text_format=text_format)

        empty_response_retry_count = 0
        incomplete_retry_count = 0
        parse_error_retry_count = 0
        schema_error_retry_count = 0
        validation_error_retry_count = 0
        while True:
            response = self._execute_with_retry(
                stage=stage,
                role=role,
                input_payload=input_payload,
                text_format=text_format,
                request_metadata=request_metadata,
                request=(
                    (lambda client: client.chat.completions.create(**request_kwargs))
                    if role.api_style == "chat_completions"
                    else (lambda client: client.responses.create(**request_kwargs))
                ),
            )
            response_metadata = self._build_response_metadata(response)
            response_status = coerce_optional_string(response_metadata.get("status"), default=None)
            resolved_output_text = (
                coerce_optional_string(response_metadata.get("resolved_output_text"), default="") or ""
            ).strip()
            if response_status not in (None, "completed"):
                error_type: Literal["empty_response", "incomplete"] = (
                    "incomplete" if response_status == "incomplete" else "empty_response"
                )
                error = JudgeClientError(
                    stage=stage,
                    error_type=error_type,
                    message=f"{stage} response status was {response_status!r}.",
                    details={"request": request_metadata, "response": response_metadata},
                )
                self._record_response_quality_failure(stage=stage, error=error)
                if error_type == "incomplete" and incomplete_retry_count < role.incomplete_retries:
                    self._record_metric(stage=stage, name="retries")
                    self._sleep(self._compute_response_retry_delay(incomplete_retry_count))
                    incomplete_retry_count += 1
                    continue
                if error_type == "empty_response" and empty_response_retry_count < role.empty_response_retries:
                    self._record_metric(stage=stage, name="retries")
                    self._sleep(self._compute_response_retry_delay(empty_response_retry_count))
                    empty_response_retry_count += 1
                    continue
                raise error
            if not resolved_output_text:
                error = JudgeClientError(
                    stage=stage,
                    error_type="empty_response",
                    message=f"{stage} response was empty.",
                    details={"request": request_metadata, "response": response_metadata},
                )
                self._record_response_quality_failure(stage=stage, error=error)
                if empty_response_retry_count < role.empty_response_retries:
                    self._record_metric(stage=stage, name="retries")
                    self._sleep(self._compute_response_retry_delay(empty_response_retry_count))
                    empty_response_retry_count += 1
                    continue
                raise error
            resolved_output: Any = resolved_output_text
            if output_validator is not None:
                try:
                    resolved_output = output_validator(resolved_output_text)
                except JudgeClientError as exc:
                    exc.add_context(
                        request=request_metadata,
                        response=response_metadata,
                        raw_output_text=resolved_output_text,
                    )
                    self._record_response_quality_failure(stage=stage, error=exc)
                    if exc.error_type == "parse_error" and parse_error_retry_count < role.parse_error_retries:
                        self._record_metric(stage=stage, name="retries")
                        self._sleep(self._compute_response_retry_delay(parse_error_retry_count))
                        parse_error_retry_count += 1
                        continue
                    if exc.error_type == "schema_error" and schema_error_retry_count < role.schema_error_retries:
                        self._record_metric(stage=stage, name="retries")
                        self._sleep(self._compute_response_retry_delay(schema_error_retry_count))
                        schema_error_retry_count += 1
                        continue
                    if (
                        exc.error_type == "validation_error"
                        and validation_error_retry_count < role.validation_error_retries
                    ):
                        self._record_metric(stage=stage, name="retries")
                        self._sleep(self._compute_response_retry_delay(validation_error_retry_count))
                        validation_error_retry_count += 1
                        continue
                    raise
            self._after_request_success(stage=stage)
            return resolved_output

    def _acquire_inflight_request(self, request_fingerprint: str) -> tuple[Future[Any], bool]:
        with self._shared_resources.inflight_requests_lock:
            existing_future = self._shared_resources.inflight_requests.get(request_fingerprint)
            if existing_future is not None:
                return existing_future, False
            created_future: Future[Any] = Future()
            self._shared_resources.inflight_requests[request_fingerprint] = created_future
            return created_future, True

    def _await_inflight_result(self, future: Future[Any]) -> Any:
        try:
            return future.result()
        except JudgeClientError as exc:
            raise exc.clone() from exc

    def _complete_inflight_request(
        self,
        *,
        request_fingerprint: str,
        future: Future[Any],
        result: Any | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._shared_resources.inflight_requests_lock:
            existing_future = self._shared_resources.inflight_requests.get(request_fingerprint)
            if existing_future is future:
                self._shared_resources.inflight_requests.pop(request_fingerprint, None)
        if future.done():
            return
        try:
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)
        except InvalidStateError:
            pass

    def _record_metric(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        name: str,
        value: float = 1.0,
    ) -> None:
        runtime = self._stage_runtime(stage)
        with self._metrics_lock:
            runtime.metrics[name] = runtime.metrics.get(name, 0.0) + value

    def _before_request(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
    ) -> None:
        runtime = self._stage_runtime(stage)
        if not runtime.circuit_breaker.allow_request():
            self._record_metric(stage=stage, name="circuit_open_rejections")
            raise JudgeClientError(
                stage=stage,
                error_type="circuit_open",
                message=f"{stage} provider circuit breaker is open; request rejected during cooldown.",
                retriable=False,
            )

        waited_seconds = 0.0
        if self._shared_resources.rpm_limiter is not None:
            waited_seconds += self._shared_resources.rpm_limiter.acquire(1.0)
        if self._shared_resources.tpm_limiter is not None:
            estimated_tokens = float(
                self._estimate_request_tokens(
                    role=role,
                    input_payload=input_payload,
                    text_format=text_format,
                )
            )
            waited_seconds += self._shared_resources.tpm_limiter.acquire(estimated_tokens)
            self._record_metric(stage=stage, name="estimated_tokens", value=estimated_tokens)
        if waited_seconds > 0:
            self._record_metric(stage=stage, name="rate_limit_wait_count")
            self._record_metric(stage=stage, name="rate_limit_wait_seconds", value=waited_seconds)
        self._record_metric(stage=stage, name="requests_started")

    def _after_request_success(self, *, stage: Literal["teacher", "rubricator", "verifier"]) -> None:
        runtime = self._stage_runtime(stage)
        runtime.circuit_breaker.record_success()
        self._record_metric(stage=stage, name="requests_succeeded")

    def _after_request_failure(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        error: JudgeClientError,
    ) -> None:
        runtime = self._stage_runtime(stage)
        self._record_metric(stage=stage, name="requests_failed")
        if error.error_type == "timeout":
            self._record_metric(stage=stage, name="timeout_errors")
        if error.retriable:
            error_snapshot = self._build_retriable_error_snapshot(error)
            if runtime.first_retriable_error is None:
                runtime.first_retriable_error = error_snapshot
            runtime.last_retriable_error = error_snapshot
            runtime.circuit_breaker.record_retriable_error()
            self._record_metric(stage=stage, name="retriable_errors")
            return
        runtime.circuit_breaker.record_ignored_failure()

    @staticmethod
    def _build_retriable_error_snapshot(error: JudgeClientError) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "stage": error.stage,
            "error_type": error.error_type,
            "retriable": error.retriable,
            "message": str(error),
        }
        if error.status_code is not None:
            snapshot["status_code"] = error.status_code
        if error.details:
            snapshot["details"] = copy.deepcopy(error.details)
        return snapshot

    def _record_response_quality_failure(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        error: JudgeClientError,
    ) -> None:
        """Record a response-quality failure (empty/incomplete) without affecting the circuit breaker."""
        runtime = self._stage_runtime(stage)
        runtime.circuit_breaker.record_response_quality_failure()
        self._record_metric(stage=stage, name="requests_failed")

    def _estimate_request_tokens(
        self,
        *,
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
    ) -> int:
        def _serialize(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            except (TypeError, ValueError):
                return str(value)

        prompt_char_count = len(_serialize(input_payload)) + len(_serialize(text_format))
        prompt_token_estimate = max(1, math.ceil(prompt_char_count / 4))
        completion_token_estimate = role.max_output_tokens if role.max_output_tokens is not None else 1024
        return max(1, prompt_token_estimate + int(completion_token_estimate))

    def _build_request_fingerprint(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
    ) -> str | None:
        fingerprint_payload = {
            "request_fingerprint_schema_version": REQUEST_FINGERPRINT_SCHEMA_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "stage": stage,
            "provider": role.provider,
            "api_style": role.api_style,
            "model": role.model,
            "base_url": role.base_url,
            "reasoning_effort": role.reasoning_effort,
            "max_output_tokens": role.max_output_tokens,
            "temperature": role.temperature,
            "top_p": role.top_p,
            "input_payload": input_payload,
            "text_format": text_format,
        }
        try:
            canonical_json = json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def _build_request_kwargs(
        self,
        *,
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if role.api_style == "chat_completions":
            request_kwargs: dict[str, Any] = {
                "model": role.model,
                "messages": self._normalize_chat_messages(input_payload),
            }
            if role.max_output_tokens is not None:
                request_kwargs["max_tokens"] = role.max_output_tokens
            if role.reasoning_effort in (None, "none") and role.temperature is not None:
                request_kwargs["temperature"] = role.temperature
            if role.reasoning_effort in (None, "none") and role.top_p is not None:
                request_kwargs["top_p"] = role.top_p
            if text_format is not None:
                request_kwargs["response_format"] = {"type": "json_object"}
            return request_kwargs

        request_kwargs: dict[str, Any] = {
            "model": role.model,
            "input": input_payload,
        }
        if role.reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": role.reasoning_effort}
        if role.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = role.max_output_tokens
        if role.reasoning_effort in (None, "none") and role.temperature is not None:
            request_kwargs["temperature"] = role.temperature
        if role.reasoning_effort in (None, "none") and role.top_p is not None:
            request_kwargs["top_p"] = role.top_p
        if text_format is not None:
            request_kwargs["text"] = {"format": text_format}
        return request_kwargs

    def _build_request_metadata(
        self,
        *,
        role: OpenAIRoleConfig,
        text_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request_metadata: dict[str, Any] = {
            "provider": role.provider,
            "api_style": role.api_style,
            "model": role.model,
            "timeout_seconds": role.timeout_seconds,
            "reasoning_effort": role.reasoning_effort,
            "max_output_tokens": role.max_output_tokens,
            "temperature": role.temperature,
            "top_p": role.top_p,
            "response_retry": {
                "empty_response_retries": role.empty_response_retries,
                "incomplete_retries": role.incomplete_retries,
                "parse_error_retries": role.parse_error_retries,
                "schema_error_retries": role.schema_error_retries,
            },
        }
        if text_format is not None:
            request_metadata["text_format"] = {
                "type": text_format.get("type"),
                "name": text_format.get("name"),
            }
        return request_metadata

    def _build_response_metadata(self, response: Any) -> dict[str, Any]:
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            finish_reason = self._to_json_compatible(getattr(first_choice, "finish_reason", None))
            resolved_output_text = self._extract_chat_completion_text(
                None if message is None else getattr(message, "content", None)
            )
            response_status = "completed"
            incomplete_details = None
            if finish_reason not in (None, "stop"):
                if finish_reason == "length":
                    response_status = "incomplete"
                    incomplete_details = {"reason": "length"}
                else:
                    response_status = str(finish_reason)
            response_metadata = {
                "id": self._to_json_compatible(getattr(response, "id", None)),
                "model": self._to_json_compatible(getattr(response, "model", None)),
                "status": response_status,
                "incomplete_details": incomplete_details,
                "usage": self._to_json_compatible(getattr(response, "usage", None)),
                "finish_reason": finish_reason,
                "resolved_output_text": resolved_output_text or None,
                "resolved_output_text_length": len(resolved_output_text),
                "resolved_output_text_source": "chat_completion_message" if resolved_output_text else "none",
                "choices": self._to_json_compatible(choices),
            }
            return {
                key: value
                for key, value in response_metadata.items()
                if value is not None and value != []
            }

        direct_output_text = coerce_optional_string(getattr(response, "output_text", None), default="") or ""
        output_payload = self._to_json_compatible(getattr(response, "output", None))
        text_fragments = self._collect_text_fragments(output_payload)
        reconstructed_output_text = "".join(text_fragments).strip()
        resolved_output_text = direct_output_text.strip() if direct_output_text.strip() else reconstructed_output_text
        resolved_output_text_source = "output_text" if direct_output_text.strip() else "output_fragments"
        if not resolved_output_text:
            resolved_output_text_source = "none"

        response_metadata = {
            "id": self._to_json_compatible(getattr(response, "id", None)),
            "model": self._to_json_compatible(getattr(response, "model", None)),
            "status": self._to_json_compatible(getattr(response, "status", None)),
            "incomplete_details": self._to_json_compatible(getattr(response, "incomplete_details", None)),
            "usage": self._to_json_compatible(getattr(response, "usage", None)),
            "output_text": direct_output_text,
            "output_text_length": len(direct_output_text),
            "output_text_fragments": text_fragments or None,
            "output_text_fragment_count": len(text_fragments),
            "reconstructed_output_text": reconstructed_output_text or None,
            "reconstructed_output_text_length": len(reconstructed_output_text),
            "resolved_output_text": resolved_output_text or None,
            "resolved_output_text_length": len(resolved_output_text),
            "resolved_output_text_source": resolved_output_text_source,
            "output": output_payload,
        }
        return {
            key: value
            for key, value in response_metadata.items()
            if value is not None and value != []
        }

    def _compute_response_retry_delay(self, attempt_index: int) -> float:
        """Fixed short delay for response-quality retries (not congestion-based)."""
        return min(self.transport.max_backoff_seconds, self.transport.initial_backoff_seconds * self._uniform(0.5, 1.5))

    def _normalize_chat_messages(self, input_payload: Any) -> list[dict[str, Any]]:
        if isinstance(input_payload, list | tuple):
            normalized_messages: list[dict[str, Any]] = []
            for item in input_payload:
                if isinstance(item, Mapping) and "role" in item and "content" in item:
                    normalized_messages.append({"role": str(item["role"]), "content": item["content"]})
                else:
                    normalized_messages.append({"role": "user", "content": self._stringify_chat_payload(item)})
            return normalized_messages
        if isinstance(input_payload, Mapping) and "role" in input_payload and "content" in input_payload:
            return [{"role": str(input_payload["role"]), "content": input_payload["content"]}]
        return [{"role": "user", "content": self._stringify_chat_payload(input_payload)}]

    def _stringify_chat_payload(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(self._to_json_compatible(value), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    def _extract_chat_completion_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list | tuple):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = coerce_optional_string(item.get("text"), default="") or ""
                    if text:
                        fragments.append(text)
            return "".join(fragments).strip()
        return coerce_optional_string(content, default="") or ""

    def _to_json_compatible(self, value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self._to_json_compatible(nested_value)
                for key, nested_value in value.items()
            }
        if isinstance(value, list | tuple):
            return [self._to_json_compatible(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return self._to_json_compatible(value.model_dump(mode="json"))
            except TypeError:
                return self._to_json_compatible(value.model_dump())
        if hasattr(value, "dict"):
            return self._to_json_compatible(value.dict())
        return str(value)

    def _collect_text_fragments(self, value: Any, *, allow_plain_text: bool = False) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if allow_plain_text and stripped else []
        if isinstance(value, Mapping):
            fragments: list[str] = []
            text_value = value.get("text")
            if text_value is not None:
                fragments.extend(self._collect_text_fragments(text_value, allow_plain_text=True))

            content_value = value.get("content")
            if content_value is not None:
                fragments.extend(self._collect_text_fragments(content_value, allow_plain_text=True))

            for key, nested_value in value.items():
                if key in {"text", "content"}:
                    continue
                if allow_plain_text and key == "value":
                    fragments.extend(self._collect_text_fragments(nested_value, allow_plain_text=True))
                    continue
                fragments.extend(self._collect_text_fragments(nested_value, allow_plain_text=False))
            return fragments
        if isinstance(value, list | tuple):
            fragments: list[str] = []
            for item in value:
                fragments.extend(self._collect_text_fragments(item, allow_plain_text=allow_plain_text))
            return fragments
        return []

    def _retry_after_seconds_from_headers(self, headers: Any) -> float | None:
        if headers is None:
            return None

        retry_after_value: Any = None
        if isinstance(headers, Mapping):
            retry_after_value = headers.get("retry-after") or headers.get("Retry-After")
        else:
            getter = getattr(headers, "get", None)
            if callable(getter):
                retry_after_value = getter("retry-after") or getter("Retry-After")

        if retry_after_value is None:
            return None

        try:
            return max(0.0, float(str(retry_after_value).strip()))
        except (TypeError, ValueError):
            return None

    def _resolve_retry_delay_seconds(self, *, error: JudgeClientError, attempt_index: int) -> float:
        retry_after_seconds = coerce_optional_float(error.details.get("retry_after_seconds"), default=None)
        if retry_after_seconds is not None:
            return max(0.0, retry_after_seconds)
        return self._compute_backoff_seconds(attempt_index)

    def _execute_with_retry(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
        request_metadata: dict[str, Any],
        request: Any,
    ) -> Any:
        max_attempts = self.transport.max_retries + 1
        last_error: JudgeClientError | None = None

        for attempt_index in range(max_attempts):
            try:
                self._before_request(
                    stage=stage,
                    role=role,
                    input_payload=input_payload,
                    text_format=text_format,
                )
                with self._shared_resources.semaphore:
                    response = request(self._get_client(role))
                if getattr(response, "error", None) is not None:
                    raise JudgeClientError(
                        stage=stage,
                        error_type="http_error",
                        message=f"{stage} response contained an API-side error: {response.error}",
                        retriable=False,
                        details={
                            "request": self._build_request_metadata(role=role, text_format=text_format),
                            "response": self._build_response_metadata(response),
                        },
                    )
                return response
            except JudgeClientError as exc:
                last_error = exc
            except APITimeoutError as exc:
                last_error = JudgeClientError(
                    stage=stage,
                    error_type="timeout",
                    message=str(exc),
                    retriable=True,
                    details={"request": copy.deepcopy(request_metadata)},
                )
            except APIConnectionError as exc:
                last_error = JudgeClientError(
                    stage=stage,
                    error_type="http_error",
                    message=str(exc),
                    retriable=True,
                    details={"request": copy.deepcopy(request_metadata)},
                )
            except APIStatusError as exc:
                status_code = getattr(exc, "status_code", None)
                response = getattr(exc, "response", None)
                retry_after_seconds = self._retry_after_seconds_from_headers(
                    None if response is None else response.headers
                )
                error_details: dict[str, Any] = {"request": copy.deepcopy(request_metadata)}
                if retry_after_seconds is not None:
                    error_details["retry_after_seconds"] = retry_after_seconds
                last_error = JudgeClientError(
                    stage=stage,
                    error_type="http_error",
                    message=str(exc),
                    status_code=status_code,
                    retriable=status_code in TRANSIENT_HTTP_STATUS_CODES,
                    details=error_details,
                )

            if last_error is None:
                continue
            self._after_request_failure(stage=stage, error=last_error)
            if not last_error.retriable or attempt_index == max_attempts - 1:
                raise last_error
            self._record_metric(stage=stage, name="retries")
            self._sleep(self._resolve_retry_delay_seconds(error=last_error, attempt_index=attempt_index))

        raise last_error or RuntimeError("unreachable")

    def _compute_backoff_seconds(self, attempt_index: int) -> float:
        base_delay = self.transport.initial_backoff_seconds * (self.transport.backoff_multiplier**attempt_index)
        jitter = self._uniform(0.5, 1.5)
        return min(self.transport.max_backoff_seconds, base_delay * jitter)

    def _get_client(self, role: OpenAIRoleConfig) -> Any:
        cache_key = (role.api_key, role.base_url, role.timeout_seconds)
        with self._shared_resources.client_cache_lock:
            client = self._shared_resources.client_cache.get(cache_key)
            if client is None:
                client = self._client_factory(
                    api_key=role.api_key,
                    base_url=role.base_url,
                    timeout=role.timeout_seconds,
                    max_retries=0,
                )
                self._shared_resources.client_cache[cache_key] = client
            return client




__all__ = [
    "PROVIDER_METRIC_NAMES",
    "REQUEST_FINGERPRINT_SCHEMA_VERSION",
    "SUPPORTED_ROLE_API_STYLES",
    "SUPPORTED_ROLE_PROVIDERS",
    "TRANSIENT_HTTP_STATUS_CODES",
    "OpenAICompatibleProvider",
    "SharedProviderResources",
    "StageProviderRuntime",
    "_empty_provider_metrics",
    "_sum_provider_metrics",
]
