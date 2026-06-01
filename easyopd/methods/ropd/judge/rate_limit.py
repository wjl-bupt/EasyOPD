from __future__ import annotations

import time
from threading import Lock
from typing import Any


class SyncTokenBucket:
    def __init__(
        self,
        *,
        rate_limit: float,
        max_tokens: float | None = None,
        time_fn: Any = time.monotonic,
        sleep: Any = time.sleep,
    ) -> None:
        self.rate_limit = rate_limit
        self.max_tokens = max_tokens or rate_limit
        self._time_fn = time_fn
        self._sleep = sleep
        self._tokens = self.max_tokens
        self._last_update: float | None = None
        self._lock = Lock()

    def acquire(self, num_tokens: float = 1.0) -> float:
        if num_tokens <= 0:
            return 0.0

        total_wait_seconds = 0.0

        if num_tokens > self.max_tokens:
            wait_seconds = 0.0
            with self._lock:
                now = self._time_fn()
                if self._last_update is None:
                    self._last_update = now

                elapsed = max(0.0, now - self._last_update)
                self._tokens = min(self.max_tokens, self._tokens + elapsed * self.rate_limit)
                self._last_update = now

                tokens_needed = num_tokens - self._tokens
                if tokens_needed > 0:
                    wait_seconds = tokens_needed / self.rate_limit

                self._tokens = max(-self.max_tokens, self._tokens - num_tokens)

            if wait_seconds > 0:
                total_wait_seconds += wait_seconds
                self._sleep(wait_seconds)
            return total_wait_seconds

        while True:
            wait_seconds = 0.0
            with self._lock:
                now = self._time_fn()
                if self._last_update is None:
                    self._last_update = now

                elapsed = max(0.0, now - self._last_update)
                self._tokens = min(self.max_tokens, self._tokens + elapsed * self.rate_limit)
                self._last_update = now

                if self._tokens >= num_tokens:
                    self._tokens -= num_tokens
                    return total_wait_seconds

                tokens_needed = num_tokens - self._tokens
                wait_seconds = tokens_needed / self.rate_limit

            total_wait_seconds += wait_seconds
            self._sleep(wait_seconds)


__all__ = ["SyncTokenBucket"]
