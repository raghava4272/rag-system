"""Thread-safe per-identity token-bucket request limiting."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


RATE_LIMIT_REQUESTS_ENV = "RAG_RATE_LIMIT_REQUESTS"
RATE_LIMIT_WINDOW_ENV = "RAG_RATE_LIMIT_WINDOW_SECONDS"
DEFAULT_RATE_LIMIT_REQUESTS = 30
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
MAX_RATE_LIMIT_REQUESTS = 10_000
MAX_RATE_LIMIT_WINDOW_SECONDS = 86_400


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Result of consuming one request token."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Smooth bursts per trusted identity within one API process."""

    def __init__(
        self,
        *,
        requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(requests, int) or isinstance(requests, bool):
            raise ValueError("requests must be an integer")
        if not 1 <= requests <= MAX_RATE_LIMIT_REQUESTS:
            raise ValueError(
                f"requests must be between 1 and {MAX_RATE_LIMIT_REQUESTS}"
            )
        if not isinstance(window_seconds, int) or isinstance(window_seconds, bool):
            raise ValueError("window_seconds must be an integer")
        if not 1 <= window_seconds <= MAX_RATE_LIMIT_WINDOW_SECONDS:
            raise ValueError(
                "window_seconds must be between 1 and "
                f"{MAX_RATE_LIMIT_WINDOW_SECONDS}"
            )

        self.requests = requests
        self.window_seconds = window_seconds
        self._refill_rate = requests / window_seconds
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._next_cleanup = 0.0

    def __repr__(self) -> str:
        return (
            f"TokenBucketRateLimiter(requests={self.requests}, "
            f"window_seconds={self.window_seconds}, "
            f"active_identity_count={len(self._buckets)})"
        )

    @classmethod
    def from_environment(cls) -> TokenBucketRateLimiter:
        raw_requests = os.getenv(
            RATE_LIMIT_REQUESTS_ENV,
            str(DEFAULT_RATE_LIMIT_REQUESTS),
        )
        raw_window = os.getenv(
            RATE_LIMIT_WINDOW_ENV,
            str(DEFAULT_RATE_LIMIT_WINDOW_SECONDS),
        )
        try:
            requests = int(raw_requests)
            window_seconds = int(raw_window)
            return cls(requests=requests, window_seconds=window_seconds)
        except (TypeError, ValueError) as error:
            raise RuntimeError("Rate limit configuration is invalid") from error

    def _prune_full_buckets(self, now: float) -> None:
        if now < self._next_cleanup:
            return
        stale_before = now - self.window_seconds
        self._buckets = {
            identity: bucket
            for identity, bucket in self._buckets.items()
            if bucket.updated_at > stale_before
        }
        self._next_cleanup = now + self.window_seconds

    def check(self, identity: str) -> RateLimitDecision:
        """Atomically consume one token for a trusted nonempty identity."""

        if not isinstance(identity, str) or not identity:
            raise ValueError("rate limit identity cannot be empty")

        with self._lock:
            now = self._clock()
            self._prune_full_buckets(now)
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.requests), updated_at=now)
                self._buckets[identity] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(
                    float(self.requests),
                    bucket.tokens + (elapsed * self._refill_rate),
                )
                bucket.updated_at = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return RateLimitDecision(
                    allowed=True,
                    limit=self.requests,
                    remaining=max(0, math.floor(bucket.tokens)),
                    retry_after_seconds=0,
                )

            retry_after = max(
                1,
                math.ceil((1.0 - bucket.tokens) / self._refill_rate),
            )
            return RateLimitDecision(
                allowed=False,
                limit=self.requests,
                remaining=0,
                retry_after_seconds=retry_after,
            )
