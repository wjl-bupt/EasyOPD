from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any

from easyopd.methods.ropd.judge.config import ProviderCircuitBreakerConfig


class ProviderCircuitBreaker:
    def __init__(
        self,
        config: ProviderCircuitBreakerConfig,
        *,
        time_fn: Any = time.monotonic,
    ) -> None:
        self.config = config
        self._time_fn = time_fn
        self._lock = Lock()
        self._state = "closed"
        self._opened_at = 0.0
        self._consecutive_retriable_errors = 0
        self._recent_outcomes: deque[bool] = deque(maxlen=config.rolling_window_size)
        self._half_open_in_flight = 0
        self._half_open_successes = 0

    def allow_request(self) -> bool:
        with self._lock:
            now = self._time_fn()
            if self._state == "open":
                if now - self._opened_at < self.config.cooldown_seconds:
                    return False
                self._state = "half_open"
                self._half_open_in_flight = 0
                self._half_open_successes = 0

            if self._state == "half_open":
                if self._half_open_in_flight >= self.config.half_open_probe_requests:
                    return False
                self._half_open_in_flight += 1

            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_retriable_errors = 0
            self._recent_outcomes.append(False)
            if self._state == "half_open":
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._half_open_successes += 1
                if (
                    self._half_open_in_flight == 0
                    and self._half_open_successes >= self.config.half_open_probe_requests
                ):
                    self._close_locked()

    def record_retriable_error(self) -> None:
        with self._lock:
            self._consecutive_retriable_errors += 1
            self._recent_outcomes.append(True)
            if self._state == "half_open":
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._open_locked()
                return

            rolling_error_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
            should_trip_by_rate = (
                len(self._recent_outcomes) >= self.config.rolling_window_size
                and rolling_error_rate >= self.config.rolling_error_rate
            )
            should_trip_by_consecutive = (
                self._consecutive_retriable_errors >= self.config.consecutive_retriable_errors
            )
            if should_trip_by_consecutive or should_trip_by_rate:
                self._open_locked()

    def record_ignored_failure(self) -> None:
        with self._lock:
            self._consecutive_retriable_errors = 0
            if self._state == "half_open":
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    def record_response_quality_failure(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    def _open_locked(self) -> None:
        self._state = "open"
        self._opened_at = self._time_fn()
        self._half_open_in_flight = 0
        self._half_open_successes = 0

    def _close_locked(self) -> None:
        self._state = "closed"
        self._consecutive_retriable_errors = 0
        self._recent_outcomes.clear()
        self._half_open_in_flight = 0
        self._half_open_successes = 0

__all__ = ["ProviderCircuitBreaker"]
