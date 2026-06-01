"""ROPD reward manager.

Wires the ROPD teacher/rubric/verifier pipeline into a verl reward manager and
applies the quality-gate / retry control flow that the ROPD reward path needs.
"""

import logging
import time
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from easyopd.methods.ropd.clients import ROPDClientConfig, build_ropd_client_config, build_ropd_pipeline
from easyopd.methods.ropd.pipeline import (
    ROPDGroup,
    ROPDPairResult,
    ROPDPipeline,
    ROPDRollout,
    canonicalize_raw_prompt,
    normalize_raw_prompt,
)
from verl.protocol import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn

logger = logging.getLogger(__name__)

RECOVERABLE_REWARD_ERROR_TYPES = frozenset({"timeout", "http_error", "empty_response", "incomplete"})
STEP_RECOVERABLE_REWARD_ERROR_TYPES = RECOVERABLE_REWARD_ERROR_TYPES | frozenset({"circuit_open"})


@dataclass(frozen=True, slots=True)
class ROPDRewardQualityGateConfig:
    enabled: bool = True
    max_fallback_rate: float = 3.0 / 32.0
    max_retry_rounds: int = 3
    retry_pair_concurrency: int = 2
    min_group_size_after_exclusion: int = 2
    max_dropped_pairs_per_group: int = 3
    min_effective_group_rate: float = 0.75
    max_step_judge_retry_attempts: int = 3
    step_judge_retry_initial_backoff_seconds: float = 10.0
    step_judge_retry_backoff_multiplier: float = 2.0
    step_judge_retry_max_backoff_seconds: float = 40.0


@dataclass(slots=True)
class ROPDPairRecord:
    uid: str
    pair_index: int
    rollout: ROPDRollout
    result: ROPDPairResult
    retry_count: int = 0
    final_status: str = "success"
    first_error_stage: str = ""
    first_error_type: str = ""


@dataclass(frozen=True, slots=True)
class ROPDRewardControl:
    fallback_rate_initial: float
    fallback_rate_repaired: float
    retry_round_count: int
    retried_pair_count: int
    retryable_pair_keys: tuple[tuple[str, int], ...]
    terminal_pair_keys: tuple[tuple[str, int], ...]
    update_mask: tuple[bool, ...]
    total_group_count: int
    effective_uid_count: int
    effective_group_rate: float
    effective_pair_count: int
    group_excluded_count: int
    group_excluded_uids: tuple[str, ...]
    excluded_pair_count: int
    quality_gate_stop: bool
    stop_reason: str
    fallback_rate_history: tuple[float, ...]
    effective_group_rate_history: tuple[float, ...]
    step_retry_attempt: int
    step_retry_exhausted: bool
    failed_pairs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fallback_rate_initial": self.fallback_rate_initial,
            "fallback_rate_repaired": self.fallback_rate_repaired,
            "retry_round_count": self.retry_round_count,
            "retried_pair_count": self.retried_pair_count,
            "retryable_pair_keys": list(self.retryable_pair_keys),
            "terminal_pair_keys": list(self.terminal_pair_keys),
            "update_mask": list(self.update_mask),
            "total_group_count": self.total_group_count,
            "effective_uid_count": self.effective_uid_count,
            "effective_group_rate": self.effective_group_rate,
            "effective_pair_count": self.effective_pair_count,
            "group_excluded_count": self.group_excluded_count,
            "group_excluded_uids": list(self.group_excluded_uids),
            "excluded_pair_count": self.excluded_pair_count,
            "quality_gate_stop": self.quality_gate_stop,
            "stop_reason": self.stop_reason,
            "fallback_rate_history": list(self.fallback_rate_history),
            "effective_group_rate_history": list(self.effective_group_rate_history),
            "step_retry_attempt": self.step_retry_attempt,
            "step_retry_exhausted": self.step_retry_exhausted,
            "failed_pairs": [dict(item) for item in self.failed_pairs],
        }


class ROPDRewardManager(AbstractRewardManager):
    EXTRA_INFO_DEFAULTS = {
        "pair_index": None,
        "group_size": None,
        "teacher_score": None,
        "student_score": None,
        "reward_gap": None,
        "student_win": False,
        "fallback_used": False,
        "judge_error": False,
        "group_concurrency_limited": False,
        "pair_concurrency_limited": False,
        "verifier_subject_concurrency_limited": False,
        "local_concurrency_limit_hit": False,
    }

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        *,
        pipeline: ROPDPipeline | None = None,
        teacher_client: Any | None = None,
        rubric_client: Any | None = None,
        verifier_client: Any | None = None,
        ropd: ROPDClientConfig | dict[str, Any] | None = None,
        client_config: ROPDClientConfig | dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        provided_count = sum(client is not None for client in (teacher_client, rubric_client, verifier_client))
        resolved_client_config = (
            self._resolve_client_config(ropd=ropd, client_config=client_config)
            if pipeline is None and provided_count == 0
            else None
        )
        self.max_group_concurrency = self._resolve_max_group_concurrency(
            ropd=ropd,
            client_config=client_config,
        )
        self.reward_quality_gate = self._resolve_reward_quality_gate_config(
            ropd=ropd,
            client_config=client_config,
        )
        self.pipeline = self._build_pipeline(
            pipeline=pipeline,
            teacher_client=teacher_client,
            rubric_client=rubric_client,
            verifier_client=verifier_client,
            resolved_client_config=resolved_client_config,
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            if return_dict:
                assert isinstance(reward_from_rm_scores, dict)
                reward_extra_info = reward_from_rm_scores.get("reward_extra_info", {})
                return {
                    "reward_tensor": reward_from_rm_scores["reward_tensor"],
                    "reward_extra_info": {
                        key: reward_extra_info[key] for key in self.EXTRA_INFO_DEFAULTS if key in reward_extra_info
                    },
                    "reward_control": self._attach_runtime_metrics(
                        self._build_passthrough_reward_control(batch_size=len(data))
                    ),
                }
            return reward_from_rm_scores

        reset_retry_state = getattr(self.pipeline, "reset_retry_state", None)
        if callable(reset_retry_state):
            reset_retry_state()

        groups = self._build_groups(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = self._init_reward_extra_info(batch_size=len(data))
        group_concurrency_limited = len(groups) > self.max_group_concurrency

        teacher_responses, pair_records = self._evaluate_groups_with_records(groups)
        reward_control = self._run_reward_quality_gate(
            groups=groups,
            pair_records=pair_records,
            teacher_responses=teacher_responses,
            batch_size=len(data),
        )
        reward_control = self._attach_runtime_metrics(reward_control)

        for group in groups:
            for rollout in group.rollouts:
                record = pair_records[(group.uid, self._pair_index_for_rollout(group, rollout.batch_index))]
                pair_result = record.result
                reward_tensor[rollout.batch_index, rollout.response_length - 1] = pair_result.reward
                reward_extra_info["pair_index"][rollout.batch_index] = pair_result.pair_index
                reward_extra_info["group_size"][rollout.batch_index] = len(group.rollouts)
                reward_extra_info["teacher_score"][rollout.batch_index] = pair_result.teacher_score
                reward_extra_info["student_score"][rollout.batch_index] = pair_result.student_score
                reward_extra_info["reward_gap"][rollout.batch_index] = pair_result.reward
                reward_extra_info["student_win"][rollout.batch_index] = pair_result.student_win
                reward_extra_info["fallback_used"][rollout.batch_index] = pair_result.fallback_used
                reward_extra_info["judge_error"][rollout.batch_index] = pair_result.judge_error
                reward_extra_info["group_concurrency_limited"][rollout.batch_index] = group_concurrency_limited
                reward_extra_info["pair_concurrency_limited"][rollout.batch_index] = (
                    pair_result.pair_concurrency_limited
                )
                reward_extra_info["verifier_subject_concurrency_limited"][rollout.batch_index] = (
                    pair_result.verifier_subject_concurrency_limited
                )
                reward_extra_info["local_concurrency_limit_hit"][rollout.batch_index] = (
                    group_concurrency_limited
                    or pair_result.pair_concurrency_limited
                    or pair_result.verifier_subject_concurrency_limited
                )

        self._validate_reward_extra_info(reward_extra_info)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "reward_control": reward_control,
            }
        return reward_tensor

    def _resolve_client_config(
        self,
        *,
        ropd: ROPDClientConfig | dict[str, Any] | None,
        client_config: ROPDClientConfig | dict[str, Any] | None,
    ) -> ROPDClientConfig | None:
        resolved_client_source = ropd if ropd is not None else client_config
        if resolved_client_source is None:
            return None
        if isinstance(resolved_client_source, ROPDClientConfig):
            return resolved_client_source
        return build_ropd_client_config(resolved_client_source)

    def _resolve_max_group_concurrency(
        self,
        *,
        ropd: ROPDClientConfig | dict[str, Any] | None,
        client_config: ROPDClientConfig | dict[str, Any] | None,
    ) -> int:
        resolved_client_source = ropd if ropd is not None else client_config
        if isinstance(resolved_client_source, ROPDClientConfig):
            return resolved_client_source.max_group_concurrency
        if isinstance(resolved_client_source, Mapping):
            return max(1, int(resolved_client_source.get("max_group_concurrency", 1)))
        return 1

    def _resolve_reward_quality_gate_config(
        self,
        *,
        ropd: ROPDClientConfig | dict[str, Any] | None,
        client_config: ROPDClientConfig | dict[str, Any] | None,
    ) -> ROPDRewardQualityGateConfig:
        resolved_client_source = ropd if ropd is not None else client_config
        gate_config = {}
        if isinstance(resolved_client_source, Mapping):
            gate_config = dict(resolved_client_source.get("reward_quality_gate", {}))

        max_fallback_rate = float(gate_config.get("max_fallback_rate", 3.0 / 32.0))
        if not 0.0 <= max_fallback_rate <= 1.0:
            raise ValueError("ropd.reward_quality_gate.max_fallback_rate must be in [0, 1].")

        max_retry_rounds = int(gate_config.get("max_retry_rounds", 3))
        if max_retry_rounds < 0:
            raise ValueError("ropd.reward_quality_gate.max_retry_rounds must be non-negative.")

        retry_pair_concurrency = int(gate_config.get("retry_pair_concurrency", 2))
        if retry_pair_concurrency < 1:
            raise ValueError("ropd.reward_quality_gate.retry_pair_concurrency must be at least 1.")

        min_group_size_after_exclusion = int(gate_config.get("min_group_size_after_exclusion", 2))
        if min_group_size_after_exclusion < 1:
            raise ValueError("ropd.reward_quality_gate.min_group_size_after_exclusion must be at least 1.")

        max_dropped_pairs_per_group = int(gate_config.get("max_dropped_pairs_per_group", 3))
        if max_dropped_pairs_per_group < 0:
            raise ValueError("ropd.reward_quality_gate.max_dropped_pairs_per_group must be non-negative.")

        min_effective_group_rate = float(gate_config.get("min_effective_group_rate", 0.75))
        if not 0.0 <= min_effective_group_rate <= 1.0:
            raise ValueError("ropd.reward_quality_gate.min_effective_group_rate must be in [0, 1].")

        max_step_judge_retry_attempts = int(gate_config.get("max_step_judge_retry_attempts", 3))
        if max_step_judge_retry_attempts < 1:
            raise ValueError("ropd.reward_quality_gate.max_step_judge_retry_attempts must be at least 1.")

        step_judge_retry_initial_backoff_seconds = float(
            gate_config.get("step_judge_retry_initial_backoff_seconds", 10.0)
        )
        if step_judge_retry_initial_backoff_seconds < 0.0:
            raise ValueError(
                "ropd.reward_quality_gate.step_judge_retry_initial_backoff_seconds must be non-negative."
            )

        step_judge_retry_backoff_multiplier = float(gate_config.get("step_judge_retry_backoff_multiplier", 2.0))
        if step_judge_retry_backoff_multiplier <= 0.0:
            raise ValueError(
                "ropd.reward_quality_gate.step_judge_retry_backoff_multiplier must be greater than 0."
            )

        step_judge_retry_max_backoff_seconds = float(gate_config.get("step_judge_retry_max_backoff_seconds", 40.0))
        if step_judge_retry_max_backoff_seconds < 0.0:
            raise ValueError(
                "ropd.reward_quality_gate.step_judge_retry_max_backoff_seconds must be non-negative."
            )
        if step_judge_retry_max_backoff_seconds < step_judge_retry_initial_backoff_seconds:
            raise ValueError(
                "ropd.reward_quality_gate.step_judge_retry_max_backoff_seconds must be at least "
                "step_judge_retry_initial_backoff_seconds."
            )

        return ROPDRewardQualityGateConfig(
            enabled=bool(gate_config.get("enabled", True)),
            max_fallback_rate=max_fallback_rate,
            max_retry_rounds=max_retry_rounds,
            retry_pair_concurrency=retry_pair_concurrency,
            min_group_size_after_exclusion=min_group_size_after_exclusion,
            max_dropped_pairs_per_group=max_dropped_pairs_per_group,
            min_effective_group_rate=min_effective_group_rate,
            max_step_judge_retry_attempts=max_step_judge_retry_attempts,
            step_judge_retry_initial_backoff_seconds=step_judge_retry_initial_backoff_seconds,
            step_judge_retry_backoff_multiplier=step_judge_retry_backoff_multiplier,
            step_judge_retry_max_backoff_seconds=step_judge_retry_max_backoff_seconds,
        )

    def _build_pipeline(
        self,
        *,
        pipeline: ROPDPipeline | None,
        teacher_client: Any | None,
        rubric_client: Any | None,
        verifier_client: Any | None,
        resolved_client_config: ROPDClientConfig | None,
    ) -> ROPDPipeline:
        if pipeline is not None:
            return pipeline

        provided_clients = [teacher_client, rubric_client, verifier_client]
        provided_count = sum(client is not None for client in provided_clients)
        if provided_count == 0:
            return build_ropd_pipeline(resolved_client_config or build_ropd_client_config())
        if provided_count != 3:
            raise ValueError(
                "ROPDRewardManager requires either `pipeline`, no explicit clients, or all of "
                "`teacher_client`, `rubric_client`, and `verifier_client`."
            )

        return ROPDPipeline(
            teacher_client=teacher_client,
            rubric_client=rubric_client,
            verifier_client=verifier_client,
        )

    def _evaluate_groups_with_records(
        self, groups: tuple[ROPDGroup, ...]
    ) -> tuple[dict[str, str | None], dict[tuple[str, int], ROPDPairRecord]]:
        if self.max_group_concurrency == 1 or len(groups) <= 1:
            detailed_results = tuple(self.pipeline.evaluate_selected_pairs(group=group) for group in groups)
        else:
            max_workers = min(self.max_group_concurrency, len(groups))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                detailed_results = tuple(
                    executor.map(lambda group: self.pipeline.evaluate_selected_pairs(group=group), groups)
                )

        teacher_responses: dict[str, str | None] = {}
        pair_records: dict[tuple[str, int], ROPDPairRecord] = {}
        for group, (teacher_response, pair_results) in zip(groups, detailed_results, strict=True):
            self._validate_pair_results(group, pair_results)
            teacher_responses[group.uid] = teacher_response
            for rollout, pair_result in zip(group.rollouts, pair_results, strict=True):
                record = ROPDPairRecord(
                    uid=group.uid,
                    pair_index=pair_result.pair_index,
                    rollout=rollout,
                    result=pair_result,
                )
                self._apply_pair_result(record, pair_result=pair_result, increment_retry=False)
                pair_records[(group.uid, pair_result.pair_index)] = record
        return teacher_responses, pair_records

    def _run_reward_quality_gate(
        self,
        *,
        groups: tuple[ROPDGroup, ...],
        pair_records: dict[tuple[str, int], ROPDPairRecord],
        teacher_responses: dict[str, str | None],
        batch_size: int,
    ) -> dict[str, Any]:
        fallback_rate_initial = self._compute_failure_rate(pair_records.values())
        if not self.reward_quality_gate.enabled:
            total_group_count = len(groups)
            effective_group_rate = self._compute_effective_group_rate(
                effective_group_count=total_group_count,
                total_group_count=total_group_count,
            )
            return ROPDRewardControl(
                fallback_rate_initial=fallback_rate_initial,
                fallback_rate_repaired=fallback_rate_initial,
                retry_round_count=0,
                retried_pair_count=0,
                retryable_pair_keys=tuple(),
                terminal_pair_keys=tuple(),
                update_mask=tuple(True for _ in range(batch_size)),
                total_group_count=total_group_count,
                effective_uid_count=len(groups),
                effective_group_rate=effective_group_rate,
                effective_pair_count=batch_size,
                group_excluded_count=0,
                group_excluded_uids=tuple(),
                excluded_pair_count=0,
                quality_gate_stop=False,
                stop_reason="",
                fallback_rate_history=(fallback_rate_initial,),
                effective_group_rate_history=(effective_group_rate,),
                step_retry_attempt=1,
                step_retry_exhausted=False,
                failed_pairs=tuple(),
            ).to_dict()

        fallback_rate_history = [fallback_rate_initial]
        effective_group_rate_history: list[float] = []
        retry_round_count = 0
        retried_pair_count = 0
        step_retry_attempt = 1
        total_group_count = len(groups)

        while True:
            if step_retry_attempt > 1:
                retry_candidates = self._collect_retry_candidates(
                    groups,
                    pair_records,
                    include_step_only=True,
                )
                if not retry_candidates:
                    break
                self._retry_selected_pairs_once(
                    groups=groups,
                    retry_candidates=retry_candidates,
                    teacher_responses=teacher_responses,
                    pair_records=pair_records,
                )
                retry_round_count += 1
                retried_pair_count += sum(len(items) for items in retry_candidates.values())
                fallback_rate_history.append(self._compute_failure_rate(pair_records.values()))

            local_retry_round_count = 0
            while (
                self._compute_failure_rate(pair_records.values()) > self.reward_quality_gate.max_fallback_rate
                and local_retry_round_count < self.reward_quality_gate.max_retry_rounds
            ):
                retry_candidates = self._collect_retry_candidates(
                    groups,
                    pair_records,
                    include_step_only=False,
                )
                if not retry_candidates:
                    break
                self._retry_selected_pairs_once(
                    groups=groups,
                    retry_candidates=retry_candidates,
                    teacher_responses=teacher_responses,
                    pair_records=pair_records,
                )
                local_retry_round_count += 1
                retry_round_count += 1
                retried_pair_count += sum(len(items) for items in retry_candidates.values())
                fallback_rate_history.append(self._compute_failure_rate(pair_records.values()))

            (
                update_mask,
                effective_uid_count,
                effective_pair_count,
                group_excluded_uids,
                effective_group_rate,
            ) = self._summarize_attempt(
                groups=groups,
                pair_records=pair_records,
                batch_size=batch_size,
            )
            effective_group_rate_history.append(effective_group_rate)

            if effective_group_rate >= self.reward_quality_gate.min_effective_group_rate:
                break
            if step_retry_attempt >= self.reward_quality_gate.max_step_judge_retry_attempts:
                break
            if not self._collect_retry_candidates(groups, pair_records, include_step_only=True):
                break

            self._sleep_before_step_retry(step_retry_attempt)
            step_retry_attempt += 1

        retryable_pair_keys = tuple(
            sorted((key for key, record in pair_records.items() if record.final_status == "retryable_failure"))
        )
        for record in pair_records.values():
            if record.final_status == "retryable_failure":
                record.final_status = "terminal_failure"

        fallback_rate_repaired = self._compute_failure_rate(pair_records.values())
        terminal_pair_keys = tuple(
            sorted((key for key, record in pair_records.items() if record.final_status == "terminal_failure"))
        )
        quality_gate_stop = effective_group_rate < self.reward_quality_gate.min_effective_group_rate
        stop_reason = "step_judge_retry_exhausted" if quality_gate_stop else ""

        return ROPDRewardControl(
            fallback_rate_initial=fallback_rate_initial,
            fallback_rate_repaired=fallback_rate_repaired,
            retry_round_count=retry_round_count,
            retried_pair_count=retried_pair_count,
            retryable_pair_keys=retryable_pair_keys,
            terminal_pair_keys=terminal_pair_keys,
            update_mask=tuple(update_mask),
            total_group_count=total_group_count,
            effective_uid_count=effective_uid_count,
            effective_group_rate=effective_group_rate,
            effective_pair_count=effective_pair_count,
            group_excluded_count=len(group_excluded_uids),
            group_excluded_uids=group_excluded_uids,
            excluded_pair_count=batch_size - effective_pair_count,
            quality_gate_stop=quality_gate_stop,
            stop_reason=stop_reason,
            fallback_rate_history=tuple(fallback_rate_history),
            effective_group_rate_history=tuple(effective_group_rate_history),
            step_retry_attempt=step_retry_attempt,
            step_retry_exhausted=quality_gate_stop,
            failed_pairs=self._build_failed_pairs(pair_records),
        ).to_dict()

    def _apply_pair_result(
        self,
        record: ROPDPairRecord,
        *,
        pair_result: ROPDPairResult,
        increment_retry: bool,
    ) -> None:
        if pair_result.fallback_used and not record.first_error_type:
            record.first_error_stage = pair_result.error_stage
            record.first_error_type = pair_result.error_type
        record.result = pair_result
        if increment_retry:
            record.retry_count += 1
        if not pair_result.fallback_used:
            record.final_status = "success"
            return
        record.final_status = (
            "retryable_failure" if self._is_step_retryable_pair_result(pair_result) else "terminal_failure"
        )

    def _retry_selected_pairs_once(
        self,
        *,
        groups: tuple[ROPDGroup, ...],
        retry_candidates: dict[str, tuple[tuple[int, ROPDRollout], ...]],
        teacher_responses: dict[str, str | None],
        pair_records: dict[tuple[str, int], ROPDPairRecord],
    ) -> None:
        groups_to_retry = [group for group in groups if group.uid in retry_candidates]
        if self.max_group_concurrency == 1 or len(groups_to_retry) <= 1:
            retry_results = [
                self.pipeline.evaluate_selected_pairs(
                    group=group,
                    pair_items=retry_candidates[group.uid],
                    teacher_response=teacher_responses.get(group.uid),
                    max_pair_concurrency=self.reward_quality_gate.retry_pair_concurrency,
                )
                for group in groups_to_retry
            ]
        else:
            max_workers = min(self.max_group_concurrency, len(groups_to_retry))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                retry_results = list(
                    executor.map(
                        lambda g: self.pipeline.evaluate_selected_pairs(
                            group=g,
                            pair_items=retry_candidates[g.uid],
                            teacher_response=teacher_responses.get(g.uid),
                            max_pair_concurrency=self.reward_quality_gate.retry_pair_concurrency,
                        ),
                        groups_to_retry,
                    )
                )

        for group, (teacher_response, pair_results) in zip(groups_to_retry, retry_results, strict=True):
            teacher_responses[group.uid] = teacher_response
            for pair_result in pair_results:
                record = pair_records[(group.uid, pair_result.pair_index)]
                self._apply_pair_result(record, pair_result=pair_result, increment_retry=True)

    def _summarize_attempt(
        self,
        *,
        groups: tuple[ROPDGroup, ...],
        pair_records: dict[tuple[str, int], ROPDPairRecord],
        batch_size: int,
    ) -> tuple[list[bool], int, int, tuple[str, ...], float]:
        update_mask, effective_uid_count, effective_pair_count, group_excluded_uids = self._build_update_mask(
            groups=groups,
            pair_records=pair_records,
            batch_size=batch_size,
        )
        effective_group_rate = self._compute_effective_group_rate(
            effective_group_count=effective_uid_count,
            total_group_count=len(groups),
        )
        return update_mask, effective_uid_count, effective_pair_count, group_excluded_uids, effective_group_rate

    def _collect_retry_candidates(
        self,
        groups: tuple[ROPDGroup, ...],
        pair_records: dict[tuple[str, int], ROPDPairRecord],
        *,
        include_step_only: bool,
    ) -> dict[str, tuple[tuple[int, ROPDRollout], ...]]:
        retry_candidates: dict[str, tuple[tuple[int, ROPDRollout], ...]] = {}
        predicate = (
            self._is_step_retryable_pair_record if include_step_only else self._is_round_retryable_pair_record
        )
        for group in groups:
            pair_items = tuple(
                (pair_index, rollout)
                for pair_index, rollout in enumerate(group.rollouts)
                if predicate(pair_records[(group.uid, pair_index)])
            )
            if pair_items:
                retry_candidates[group.uid] = pair_items
        return retry_candidates

    @staticmethod
    def _is_round_retryable_pair_record(record: ROPDPairRecord) -> bool:
        if record.final_status != "retryable_failure":
            return False
        return ROPDRewardManager._is_round_retryable_pair_result(record.result)

    @staticmethod
    def _is_step_retryable_pair_record(record: ROPDPairRecord) -> bool:
        if record.final_status != "retryable_failure":
            return False
        return ROPDRewardManager._is_step_retryable_pair_result(record.result)

    @staticmethod
    def _is_round_retryable_pair_result(pair_result: ROPDPairResult) -> bool:
        error_type = pair_result.error_type
        if error_type in {"parse_error", "schema_error"}:
            return pair_result.error_stage == "verifier"
        if error_type == "validation_error":
            return pair_result.error_stage in {"rubricator", "verifier"}
        if error_type not in RECOVERABLE_REWARD_ERROR_TYPES:
            return False
        if error_type != "http_error":
            return True
        retriable = pair_result.error_details.get("retriable")
        if isinstance(retriable, bool):
            return retriable
        return True

    @staticmethod
    def _is_step_retryable_pair_result(pair_result: ROPDPairResult) -> bool:
        error_type = pair_result.error_type
        if error_type in {"parse_error", "schema_error"}:
            return pair_result.error_stage == "verifier"
        if error_type == "validation_error":
            return pair_result.error_stage in {"rubricator", "verifier"}
        if error_type not in STEP_RECOVERABLE_REWARD_ERROR_TYPES:
            return False
        if error_type != "http_error":
            return True
        retriable = pair_result.error_details.get("retriable")
        if isinstance(retriable, bool):
            return retriable
        return True

    def _build_update_mask(
        self,
        *,
        groups: tuple[ROPDGroup, ...],
        pair_records: dict[tuple[str, int], ROPDPairRecord],
        batch_size: int,
    ) -> tuple[list[bool], int, int, tuple[str, ...]]:
        update_mask = [False] * batch_size
        effective_uid_count = 0
        group_excluded_uids: list[str] = []

        for group in groups:
            group_records = [pair_records[(group.uid, pair_index)] for pair_index in range(len(group.rollouts))]
            surviving_records = [record for record in group_records if record.final_status == "success"]
            dropped_records = [record for record in group_records if record.final_status != "success"]
            if len(dropped_records) > self.reward_quality_gate.max_dropped_pairs_per_group:
                logger.warning(
                    "Group uid=%s excluded: dropped_pairs=%d exceeds max_dropped_pairs_per_group=%d (total=%d)",
                    group.uid,
                    len(dropped_records),
                    self.reward_quality_gate.max_dropped_pairs_per_group,
                    len(group.rollouts),
                )
                group_excluded_uids.append(group.uid)
                continue
            if len(surviving_records) < self.reward_quality_gate.min_group_size_after_exclusion:
                logger.warning(
                    "Group uid=%s excluded: surviving_pairs=%d below min_group_size_after_exclusion=%d (total=%d)",
                    group.uid,
                    len(surviving_records),
                    self.reward_quality_gate.min_group_size_after_exclusion,
                    len(group.rollouts),
                )
                group_excluded_uids.append(group.uid)
                continue

            effective_uid_count += 1
            for record in surviving_records:
                update_mask[record.rollout.batch_index] = True

        effective_pair_count = sum(update_mask)
        return update_mask, effective_uid_count, effective_pair_count, tuple(group_excluded_uids)

    def _build_failed_pairs(
        self,
        pair_records: dict[tuple[str, int], ROPDPairRecord],
    ) -> tuple[dict[str, Any], ...]:
        failed_pairs = []
        for record in pair_records.values():
            if record.final_status == "success":
                continue
            failed_pairs.append(
                {
                    "uid": record.uid,
                    "pair_index": record.pair_index,
                    "batch_index": record.rollout.batch_index,
                    "error_stage": record.result.error_stage,
                    "error_type": record.result.error_type,
                    "first_error_stage": record.first_error_stage or record.result.error_stage,
                    "first_error_type": record.first_error_type or record.result.error_type,
                    "retry_count": record.retry_count,
                    "final_status": record.final_status,
                }
            )
        failed_pairs.sort(key=lambda item: (item["uid"], item["pair_index"]))
        return tuple(failed_pairs)

    def _attach_runtime_metrics(self, reward_control: dict[str, Any]) -> dict[str, Any]:
        runtime_snapshot = self._snapshot_runtime_metrics()
        if not runtime_snapshot:
            return reward_control
        enriched_reward_control = dict(reward_control)
        enriched_reward_control["provider_metrics"] = runtime_snapshot
        return enriched_reward_control

    def _snapshot_runtime_metrics(self) -> dict[str, Any]:
        snapshot_fn = getattr(self.pipeline, "snapshot_runtime_metrics", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn()
            if isinstance(snapshot, Mapping):
                return dict(snapshot)

        provider = getattr(getattr(self.pipeline, "teacher_client", None), "provider", None)
        provider_snapshot_fn = getattr(provider, "snapshot_metrics", None)
        if callable(provider_snapshot_fn):
            snapshot = provider_snapshot_fn()
            if isinstance(snapshot, Mapping):
                return dict(snapshot)

        return {}

    def _build_passthrough_reward_control(self, batch_size: int) -> dict[str, Any]:
        return ROPDRewardControl(
            fallback_rate_initial=0.0,
            fallback_rate_repaired=0.0,
            retry_round_count=0,
            retried_pair_count=0,
            retryable_pair_keys=tuple(),
            terminal_pair_keys=tuple(),
            update_mask=tuple(True for _ in range(batch_size)),
            total_group_count=0,
            effective_uid_count=0,
            effective_group_rate=0.0,
            effective_pair_count=batch_size,
            group_excluded_count=0,
            group_excluded_uids=tuple(),
            excluded_pair_count=0,
            quality_gate_stop=False,
            stop_reason="",
            fallback_rate_history=(0.0,),
            effective_group_rate_history=(0.0,),
            step_retry_attempt=0,
            step_retry_exhausted=False,
            failed_pairs=tuple(),
        ).to_dict()

    def _compute_step_retry_sleep_seconds(self, retry_round: int) -> float:
        if retry_round <= 0:
            return 0.0
        sleep_seconds = self.reward_quality_gate.step_judge_retry_initial_backoff_seconds * (
            self.reward_quality_gate.step_judge_retry_backoff_multiplier ** (retry_round - 1)
        )
        return min(sleep_seconds, self.reward_quality_gate.step_judge_retry_max_backoff_seconds)

    def _sleep_before_step_retry(self, retry_round: int) -> None:
        sleep_seconds = self._compute_step_retry_sleep_seconds(retry_round)
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

    @staticmethod
    def _extract_reward_from_rm_scores(data: DataProto, return_dict: bool) -> torch.Tensor | dict[str, Any] | None:
        if data.batch is None or "rm_scores" not in data.batch.keys():
            return None

        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {
                key: data.non_tensor_batch[key] for key in reward_extra_keys if key in data.non_tensor_batch
            }
            return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        return data.batch["rm_scores"]

    @staticmethod
    def _compute_effective_group_rate(*, effective_group_count: int, total_group_count: int) -> float:
        if total_group_count <= 0:
            return 0.0
        return effective_group_count / total_group_count

    @staticmethod
    def _compute_failure_rate(records: Any) -> float:
        records_list = list(records)
        if not records_list:
            return 0.0
        failure_count = sum(1 for record in records_list if record.final_status != "success")
        return failure_count / len(records_list)

    @staticmethod
    def _pair_index_for_rollout(group: ROPDGroup, batch_index: int) -> int:
        for pair_index, rollout in enumerate(group.rollouts):
            if rollout.batch_index == batch_index:
                return pair_index
        raise KeyError(f"Could not find rollout with batch_index={batch_index} in group {group.uid!r}.")

    def _build_groups(self, data: DataProto) -> tuple[ROPDGroup, ...]:
        responses = data.batch["responses"]
        response_width = responses.shape[-1]
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_width:]
        valid_response_lengths = response_mask.sum(dim=-1)

        raw_prompts = data.non_tensor_batch["raw_prompt"]
        uids = data.non_tensor_batch["uid"]

        grouped_rollouts: OrderedDict[str, list[ROPDRollout]] = OrderedDict()
        grouped_prompts: dict[str, Any] = {}
        grouped_prompt_keys: dict[str, str] = {}

        for batch_index in range(len(data)):
            uid = str(uids[batch_index])
            raw_prompt = normalize_raw_prompt(raw_prompts[batch_index])
            raw_prompt_key = canonicalize_raw_prompt(raw_prompt)
            valid_response_length = int(valid_response_lengths[batch_index].item())
            if valid_response_length <= 0:
                raise ValueError(f"Sample at batch index {batch_index} has no valid response tokens.")

            if uid in grouped_prompt_keys and grouped_prompt_keys[uid] != raw_prompt_key:
                raise ValueError(f"Found multiple raw_prompt values for the same uid={uid!r}.")

            grouped_prompt_keys.setdefault(uid, raw_prompt_key)
            grouped_prompts.setdefault(uid, raw_prompt)
            grouped_rollouts.setdefault(uid, []).append(
                ROPDRollout(
                    batch_index=batch_index,
                    response_text=self._decode_response_text(responses[batch_index], valid_response_length),
                    response_length=valid_response_length,
                )
            )

        return tuple(
            ROPDGroup(uid=uid, raw_prompt=grouped_prompts[uid], rollouts=tuple(rollouts))
            for uid, rollouts in grouped_rollouts.items()
        )

    def _decode_response_text(self, response_ids: torch.Tensor, valid_response_length: int) -> str:
        valid_response_ids = response_ids[:valid_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    def _init_reward_extra_info(self, batch_size: int) -> dict[str, list[Any]]:
        return {key: [default_value] * batch_size for key, default_value in self.EXTRA_INFO_DEFAULTS.items()}

    def _validate_pair_results(
        self,
        group: ROPDGroup,
        pair_results: tuple[ROPDPairResult, ...],
    ) -> None:
        if len(pair_results) != len(group.rollouts):
            raise ValueError(
                f"Pipeline returned {len(pair_results)} pair results for a group with {len(group.rollouts)} rollouts."
            )

        for expected_pair_index, (rollout, pair_result) in enumerate(zip(group.rollouts, pair_results, strict=True)):
            if pair_result.batch_index != rollout.batch_index:
                raise ValueError(
                    "Pipeline returned pair results out of order: "
                    f"expected batch_index={rollout.batch_index}, got {pair_result.batch_index}."
                )
            if pair_result.pair_index != expected_pair_index:
                raise ValueError(
                    "Pipeline returned pair results with non-contiguous pair_index values: "
                    f"expected {expected_pair_index}, got {pair_result.pair_index}."
                )

    def _validate_reward_extra_info(self, reward_extra_info: dict[str, list[Any]]) -> None:
        for key in ("pair_index", "group_size", "teacher_score", "student_score", "reward_gap"):
            if any(value is None for value in reward_extra_info[key]):
                raise ValueError(f"reward_extra_info[{key!r}] contains unset entries.")


__all__ = [
    "ROPDPairRecord",
    "ROPDRewardControl",
    "ROPDRewardManager",
    "ROPDRewardQualityGateConfig",
    "register_ropd_reward_manager",
]


def register_ropd_reward_manager() -> None:
    """Register `ROPDRewardManager` under the name `ropd` in verl's registry.

    Idempotent: re-registration is a no-op as long as the same class is provided.
    """
    from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY, register

    if REWARD_MANAGER_REGISTRY.get("ropd") is ROPDRewardManager:
        return
    register("ropd")(ROPDRewardManager)
