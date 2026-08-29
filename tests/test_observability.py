"""Privacy-safe structured logging contracts."""

from __future__ import annotations

import io
import json
import logging
import unittest
from uuid import UUID

from src.observability import (
    StructuredJsonFormatter,
    bind_log_context,
    normalize_request_id,
    parse_log_level,
)


class RequestIdTests(unittest.TestCase):
    def test_valid_uuid_is_canonicalized_and_preserved(self) -> None:
        request_id = "11446D6E-4948-4C80-9816-C752D1416FC5"

        resolved = normalize_request_id(request_id)

        self.assertEqual(resolved, request_id.lower())

    def test_missing_or_malformed_values_are_replaced_with_uuid4(self) -> None:
        for supplied in (None, "", "not-a-uuid", "bad-value\nforged-log-line"):
            with self.subTest(supplied=supplied):
                resolved = normalize_request_id(supplied)
                parsed = UUID(resolved)

                self.assertEqual(parsed.version, 4)
                self.assertNotEqual(resolved, supplied)


class StructuredLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = io.StringIO()
        self.handler = logging.StreamHandler(self.output)
        self.handler.setFormatter(StructuredJsonFormatter())
        self.logger = logging.getLogger(f"rag.tests.{id(self)}")
        self.logger.handlers = [self.handler]
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.handlers.clear()

    def test_logs_are_json_with_correlated_allowlisted_fields(self) -> None:
        secret = "Bearer secret-api-key"
        with bind_log_context(
            request_id="11446d6e-4948-4c80-9816-c752d1416fc5",
            user_id="test-user",
            session_id="b07e89c8-bdb9-456f-a255-718b48afbd43",
        ):
            self.logger.info(
                "request.completed",
                extra={
                    "method": "POST",
                    "path": "/v1/query",
                    "status_code": 200,
                    "duration_ms": 12.3456,
                    "question": "private question text",
                    "authorization": secret,
                },
            )

        encoded = self.output.getvalue()
        event = json.loads(encoded)
        self.assertEqual(event["event"], "request.completed")
        self.assertEqual(event["level"], "INFO")
        self.assertEqual(event["logger"], self.logger.name)
        self.assertEqual(
            event["request_id"],
            "11446d6e-4948-4c80-9816-c752d1416fc5",
        )
        self.assertEqual(event["user_id"], "test-user")
        self.assertEqual(
            event["session_id"],
            "b07e89c8-bdb9-456f-a255-718b48afbd43",
        )
        self.assertEqual(event["method"], "POST")
        self.assertEqual(event["path"], "/v1/query")
        self.assertEqual(event["status_code"], 200)
        self.assertEqual(event["duration_ms"], 12.346)
        self.assertRegex(event["timestamp"], r"Z$")
        self.assertNotIn("private question text", encoded)
        self.assertNotIn(secret, encoded)

    def test_context_is_reset_after_request_scope(self) -> None:
        with bind_log_context(request_id="first-request"):
            self.logger.info("inside")
        self.logger.info("outside")

        inside, outside = [
            json.loads(line) for line in self.output.getvalue().splitlines()
        ]
        self.assertEqual(inside["request_id"], "first-request")
        self.assertNotIn("request_id", outside)

    def test_exception_details_and_unknown_fields_are_not_serialized(self) -> None:
        try:
            raise RuntimeError("provider returned private document text")
        except RuntimeError:
            self.logger.exception(
                "request.failed",
                extra={
                    "exception_type": "RuntimeError",
                    "retrieved_chunks": "private chunk",
                },
            )

        encoded = self.output.getvalue()
        event = json.loads(encoded)
        self.assertEqual(event["exception_type"], "RuntimeError")
        self.assertNotIn("provider returned", encoded)
        self.assertNotIn("private chunk", encoded)
        self.assertNotIn("traceback", event)

    def test_log_level_parser_accepts_known_levels_and_rejects_others(self) -> None:
        self.assertEqual(parse_log_level("warning"), logging.WARNING)
        self.assertEqual(parse_log_level(None), logging.INFO)
        with self.assertRaisesRegex(ValueError, "RAG_LOG_LEVEL"):
            parse_log_level("verbose")


if __name__ == "__main__":
    unittest.main()
