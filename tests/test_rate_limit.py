"""Deterministic per-user token-bucket rate limiting tests."""

from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from src.rate_limit import TokenBucketRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TokenBucketRateLimiterTests(unittest.TestCase):
    def test_burst_is_limited_and_tokens_refill_smoothly(self) -> None:
        clock = FakeClock()
        limiter = TokenBucketRateLimiter(
            requests=2,
            window_seconds=10,
            clock=clock,
        )

        first = limiter.check("alice")
        second = limiter.check("alice")
        blocked = limiter.check("alice")

        self.assertTrue(first.allowed)
        self.assertEqual(first.remaining, 1)
        self.assertTrue(second.allowed)
        self.assertEqual(second.remaining, 0)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.retry_after_seconds, 5)

        clock.advance(5)
        recovered = limiter.check("alice")
        self.assertTrue(recovered.allowed)
        self.assertEqual(recovered.remaining, 0)

    def test_users_have_independent_buckets(self) -> None:
        limiter = TokenBucketRateLimiter(
            requests=1,
            window_seconds=60,
            clock=FakeClock(),
        )

        self.assertTrue(limiter.check("alice").allowed)
        self.assertFalse(limiter.check("alice").allowed)
        self.assertTrue(limiter.check("bob").allowed)

    def test_concurrent_requests_cannot_overdraw_one_bucket(self) -> None:
        limiter = TokenBucketRateLimiter(
            requests=5,
            window_seconds=60,
            clock=FakeClock(),
        )

        with ThreadPoolExecutor(max_workers=20) as executor:
            decisions = list(executor.map(lambda _: limiter.check("alice"), range(20)))

        self.assertEqual(sum(decision.allowed for decision in decisions), 5)
        self.assertEqual(sum(not decision.allowed for decision in decisions), 15)

    def test_configuration_is_bounded_and_identity_must_be_nonempty(self) -> None:
        for requests, window in ((0, 60), (10_001, 60), (1, 0), (1, 86_401)):
            with self.subTest(requests=requests, window=window):
                with self.assertRaises(ValueError):
                    TokenBucketRateLimiter(
                        requests=requests,
                        window_seconds=window,
                    )

        limiter = TokenBucketRateLimiter(requests=1, window_seconds=60)
        with self.assertRaises(ValueError):
            limiter.check("")

    def test_environment_configuration_has_safe_defaults_and_rejects_invalid_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            limiter = TokenBucketRateLimiter.from_environment()

        decision = limiter.check("alice")
        self.assertEqual(decision.limit, 30)

        with patch.dict(
            os.environ,
            {
                "RAG_RATE_LIMIT_REQUESTS": "not-an-integer",
                "RAG_RATE_LIMIT_WINDOW_SECONDS": "60",
            },
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                TokenBucketRateLimiter.from_environment()


if __name__ == "__main__":
    unittest.main()
