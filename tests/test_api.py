"""HTTP contract tests for the RAG API boundary."""

from __future__ import annotations

import io
import json
import logging
import unittest
from uuid import UUID

from fastapi.testclient import TestClient

from src.api import create_app
from src.auth import APIKeyAuthenticator, hash_api_key
from src.generate import Citation, GeneratedAnswer, NO_CONTEXT_ANSWER
from src.observability import StructuredJsonFormatter
from src.permissions import DocumentPermissions
from src.rate_limit import TokenBucketRateLimiter


TEST_API_KEY = "test-api-key-with-at-least-thirty-two-characters"
TEST_USER_ID = "test-user"
TEST_ALLOWED_SOURCES = frozenset({"data/raw/guide.md"})


def test_authenticator() -> APIKeyAuthenticator:
    return APIKeyAuthenticator({TEST_USER_ID: hash_api_key(TEST_API_KEY)})


def test_permissions() -> DocumentPermissions:
    return DocumentPermissions({TEST_USER_ID: sorted(TEST_ALLOWED_SOURCES)})


def test_rate_limiter(*, requests: int = 100) -> TokenBucketRateLimiter:
    return TokenBucketRateLimiter(
        requests=requests,
        window_seconds=60,
        clock=lambda: 100.0,
    )


def auth_headers(*, session_id: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    if session_id is not None:
        headers["X-Session-ID"] = session_id
    return headers


def test_app(*, pipeline: object, **kwargs: object):
    return create_app(
        pipeline=pipeline,
        authenticator=test_authenticator(),
        document_permissions=test_permissions(),
        rate_limiter=test_rate_limiter(),
        **kwargs,
    )


class FakePipeline:
    def __init__(
        self,
        answer: GeneratedAnswer | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer or GeneratedAnswer(
            text="Hybrid retrieval combines semantic and keyword evidence [1].",
            citations=[
                Citation(
                    number=1,
                    record_id="guide.md:0",
                    source="data/raw/guide.md",
                    page_number=None,
                    chunk_index=0,
                )
            ],
        )
        self.error = error
        self.calls: list[tuple[str, int, frozenset[str]]] = []

    def ask(
        self,
        question: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str],
    ) -> GeneratedAnswer:
        self.calls.append((question, top_k, allowed_sources))
        if self.error is not None:
            raise self.error
        return self.answer


class ApiHealthTests(unittest.TestCase):
    def test_every_response_gets_a_safe_request_id(self) -> None:
        with TestClient(test_app(pipeline=FakePipeline())) as client:
            generated = client.get("/health/live")
            preserved = client.get(
                "/health/live",
                headers={
                    "X-Request-ID": "11446D6E-4948-4C80-9816-C752D1416FC5"
                },
            )
            replaced = client.get(
                "/health/live",
                headers={"X-Request-ID": "bad-value\nforged-log-line"},
            )

        self.assertEqual(UUID(generated.headers["x-request-id"]).version, 4)
        self.assertEqual(
            preserved.headers["x-request-id"],
            "11446d6e-4948-4c80-9816-c752d1416fc5",
        )
        self.assertEqual(UUID(replaced.headers["x-request-id"]).version, 4)
        self.assertNotIn("forged", replaced.headers["x-request-id"])

    def test_validation_errors_are_also_correlated(self) -> None:
        with TestClient(test_app(pipeline=FakePipeline())) as client:
            response = client.post(
                "/v1/query",
                json={"question": ""},
                headers=auth_headers(),
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(UUID(response.headers["x-request-id"]).version, 4)

    def test_liveness_and_readiness_are_distinct(self) -> None:
        with TestClient(test_app(pipeline=FakePipeline())) as client:
            live_response = client.get("/health/live")
            ready_response = client.get("/health/ready")

        self.assertEqual(live_response.status_code, 200)
        self.assertEqual(live_response.json(), {"status": "alive"})
        self.assertEqual(ready_response.status_code, 200)
        self.assertEqual(ready_response.json(), {"status": "ready"})

    def test_failed_startup_stays_live_but_is_not_ready(self) -> None:
        def fail_to_build() -> FakePipeline:
            raise RuntimeError("GROQ_API_KEY accidentally leaked in this message")

        with TestClient(
            create_app(
                pipeline_factory=fail_to_build,
                authenticator=test_authenticator(),
                document_permissions=test_permissions(),
            )
        ) as client:
            live_response = client.get("/health/live")
            ready_response = client.get("/health/ready")
            query_response = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
            )

        self.assertEqual(live_response.status_code, 200)
        self.assertEqual(ready_response.status_code, 503)
        self.assertEqual(ready_response.json(), {"status": "not_ready"})
        self.assertEqual(query_response.status_code, 503)
        self.assertNotIn("GROQ_API_KEY", query_response.text)

    def test_pipeline_factory_runs_once_per_application_lifetime(self) -> None:
        pipeline = FakePipeline()
        factory_calls: list[None] = []

        def build_pipeline() -> FakePipeline:
            factory_calls.append(None)
            return pipeline

        with TestClient(
            create_app(
                pipeline_factory=build_pipeline,
                authenticator=test_authenticator(),
                document_permissions=test_permissions(),
            )
        ) as client:
            client.post(
                "/v1/query",
                json={"question": "First?"},
                headers=auth_headers(),
            )
            client.post(
                "/v1/query",
                json={"question": "Second?"},
                headers=auth_headers(),
            )

        self.assertEqual(factory_calls, [None])


class ApiQueryTests(unittest.TestCase):
    def test_request_log_is_correlated_without_sensitive_content(self) -> None:
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(StructuredJsonFormatter())
        logger = logging.getLogger(f"rag.api-test.{id(self)}")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        private_question = "What does private-project-codename mean?"
        private_answer = "private-project-codename is confidential [1]."
        pipeline = FakePipeline(
            GeneratedAnswer(
                text=private_answer,
                citations=[
                    Citation(
                        number=1,
                        record_id="guide.md:0",
                        source="data/raw/guide.md",
                        page_number=None,
                        chunk_index=0,
                    )
                ],
            )
        )
        session_id = "b07e89c8-bdb9-456f-a255-718b48afbd43"
        request_id = "11446d6e-4948-4c80-9816-c752d1416fc5"

        try:
            with TestClient(
                test_app(pipeline=pipeline, request_logger=logger)
            ) as client:
                response = client.post(
                    "/v1/query?ignored=private-query-string",
                    json={"question": private_question},
                    headers={
                        **auth_headers(session_id=session_id),
                        "X-Request-ID": request_id,
                    },
                )
        finally:
            logger.handlers.clear()

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        completed = next(
            event for event in events if event["event"] == "request.completed"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(completed["request_id"], request_id)
        self.assertEqual(completed["user_id"], TEST_USER_ID)
        self.assertEqual(completed["session_id"], session_id)
        self.assertEqual(completed["method"], "POST")
        self.assertEqual(completed["path"], "/v1/query")
        self.assertEqual(completed["status_code"], 200)
        self.assertGreaterEqual(completed["duration_ms"], 0)
        encoded = output.getvalue()
        for private_value in (
            TEST_API_KEY,
            private_question,
            private_answer,
            "private-query-string",
            "Authorization",
        ):
            self.assertNotIn(private_value, encoded)

    def test_query_returns_a_typed_answer_and_structured_citations(self) -> None:
        pipeline = FakePipeline()

        with TestClient(test_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "  How does hybrid retrieval work?  ", "top_k": 3},
                headers=auth_headers(
                    session_id="11446d6e-4948-4c80-9816-c752d1416fc5"
                ),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            pipeline.calls,
            [("How does hybrid retrieval work?", 3, TEST_ALLOWED_SOURCES)],
        )
        self.assertEqual(
            response.json(),
            {
                "answer": pipeline.answer.text,
                "abstained": False,
                "user_id": TEST_USER_ID,
                "session_id": "11446d6e-4948-4c80-9816-c752d1416fc5",
                "citations": [
                    {
                        "number": 1,
                        "record_id": "guide.md:0",
                        "source": "data/raw/guide.md",
                        "page_number": None,
                        "chunk_index": 0,
                    }
                ],
            },
        )
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(
            response.headers["x-session-id"],
            "11446d6e-4948-4c80-9816-c752d1416fc5",
        )

    def test_no_context_answer_is_an_explicit_abstention(self) -> None:
        pipeline = FakePipeline(
            GeneratedAnswer(text=NO_CONTEXT_ANSWER, citations=[]),
        )

        with TestClient(test_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is outside the corpus?"},
                headers=auth_headers(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["abstained"])
        self.assertEqual(response.json()["citations"], [])

    def test_invalid_requests_are_rejected_before_calling_the_pipeline(self) -> None:
        pipeline = FakePipeline()
        invalid_payloads = (
            {"question": "   "},
            {"question": "Valid question", "top_k": 0},
            {"question": "Valid question", "top_k": 21},
            {"question": "Valid question", "unexpected": True},
        )

        with TestClient(test_app(pipeline=pipeline)) as client:
            responses = [
                client.post("/v1/query", json=payload, headers=auth_headers())
                for payload in invalid_payloads
            ]

        self.assertTrue(all(response.status_code == 422 for response in responses))
        self.assertEqual(pipeline.calls, [])

    def test_pipeline_failures_return_a_safe_stable_error(self) -> None:
        pipeline = FakePipeline(
            error=RuntimeError("provider response contained a private detail"),
        )

        with TestClient(test_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=auth_headers(),
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "rag_unavailable",
                    "message": "The RAG service is temporarily unavailable.",
                }
            },
        )
        self.assertNotIn("private detail", response.text)


class ApiAuthenticationTests(unittest.TestCase):
    def test_missing_and_invalid_credentials_share_one_safe_response(self) -> None:
        pipeline = FakePipeline()

        with TestClient(test_app(pipeline=pipeline)) as client:
            missing = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
            )
            invalid = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers={"Authorization": "Bearer wrong-api-key-value"},
            )

        expected = {
            "error": {
                "code": "unauthorized",
                "message": "Valid bearer authentication is required.",
            }
        }
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(missing.json(), expected)
        self.assertEqual(invalid.json(), expected)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        self.assertEqual(invalid.headers["www-authenticate"], "Bearer")
        self.assertEqual(pipeline.calls, [])

    def test_user_identity_comes_from_the_key_not_a_spoofable_header(self) -> None:
        pipeline = FakePipeline()
        headers = auth_headers()
        headers["X-User-ID"] = "administrator"

        with TestClient(test_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=headers,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], TEST_USER_ID)
        self.assertNotEqual(response.json()["user_id"], "administrator")

    def test_server_generates_a_uuid_session_when_client_omits_one(self) -> None:
        with TestClient(test_app(pipeline=FakePipeline())) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=auth_headers(),
            )

        session_id = UUID(response.json()["session_id"])
        self.assertEqual(session_id.version, 4)
        self.assertEqual(response.headers["x-session-id"], str(session_id))

    def test_malformed_session_is_rejected_before_pipeline_execution(self) -> None:
        pipeline = FakePipeline()

        with TestClient(test_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=auth_headers(session_id="not-a-uuid"),
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(pipeline.calls, [])

    def test_missing_authentication_configuration_fails_readiness(self) -> None:
        def fail_to_build_authenticator() -> APIKeyAuthenticator:
            raise RuntimeError("raw secret must not appear in responses")

        with TestClient(
            create_app(
                pipeline=FakePipeline(),
                authenticator_factory=fail_to_build_authenticator,
                document_permissions=test_permissions(),
            )
        ) as client:
            ready = client.get("/health/ready")
            query = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=auth_headers(),
            )

        self.assertEqual(ready.status_code, 503)
        self.assertEqual(query.status_code, 503)
        self.assertNotIn("raw secret", ready.text + query.text)

    def test_openapi_marks_only_the_query_as_bearer_protected(self) -> None:
        with TestClient(test_app(pipeline=FakePipeline())) as client:
            schema = client.get("/openapi.json").json()

        security_schemes = schema["components"]["securitySchemes"]
        self.assertEqual(security_schemes["HTTPBearer"]["scheme"], "bearer")
        self.assertIn("security", schema["paths"]["/v1/query"]["post"])
        self.assertNotIn("security", schema["paths"]["/health/live"]["get"])


class ApiDocumentAuthorizationTests(unittest.TestCase):
    def test_authenticated_identity_selects_the_retrieval_scope(self) -> None:
        pipeline = FakePipeline()

        with TestClient(test_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=auth_headers(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            pipeline.calls,
            [("What is RAG?", 5, TEST_ALLOWED_SOURCES)],
        )

    def test_authenticated_user_without_a_permission_rule_is_forbidden(self) -> None:
        pipeline = FakePipeline()
        permissions = DocumentPermissions({"different-user": ["data/raw/guide.md"]})

        with TestClient(
            create_app(
                pipeline=pipeline,
                authenticator=test_authenticator(),
                document_permissions=permissions,
            )
        ) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=auth_headers(),
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "forbidden",
                    "message": "The authenticated user cannot access this resource.",
                }
            },
        )
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(pipeline.calls, [])

    def test_missing_permission_configuration_fails_readiness_safely(self) -> None:
        def fail_to_build_permissions() -> DocumentPermissions:
            raise RuntimeError("private source names must not reach responses")

        with TestClient(
            create_app(
                pipeline=FakePipeline(),
                authenticator=test_authenticator(),
                document_permissions_factory=fail_to_build_permissions,
            )
        ) as client:
            ready = client.get("/health/ready")
            query = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=auth_headers(),
            )

        self.assertEqual(ready.status_code, 503)
        self.assertEqual(query.status_code, 503)
        self.assertNotIn("private source", ready.text + query.text)


class ApiRateLimitTests(unittest.TestCase):
    def test_per_user_limit_blocks_pipeline_and_returns_retry_headers(self) -> None:
        pipeline = FakePipeline()
        limiter = test_rate_limiter(requests=2)
        bypass_headers = auth_headers(
            session_id="d31cb570-0dd2-40eb-95f6-c753b647bc89"
        )
        bypass_headers["X-User-ID"] = "different-user"

        with TestClient(
            create_app(
                pipeline=pipeline,
                authenticator=test_authenticator(),
                document_permissions=test_permissions(),
                rate_limiter=limiter,
            )
        ) as client:
            first = client.post(
                "/v1/query",
                json={"question": "First question?"},
                headers=auth_headers(
                    session_id="11446d6e-4948-4c80-9816-c752d1416fc5"
                ),
            )
            second = client.post(
                "/v1/query",
                json={"question": "Second question?"},
                headers=auth_headers(
                    session_id="b07e89c8-bdb9-456f-a255-718b48afbd43"
                ),
            )
            blocked = client.post(
                "/v1/query",
                json={"question": "Third question?"},
                headers=bypass_headers,
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.headers["ratelimit-limit"], "2")
        self.assertEqual(first.headers["ratelimit-remaining"], "1")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.headers["ratelimit-remaining"], "0")
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.headers["retry-after"], "30")
        self.assertEqual(blocked.headers["ratelimit-limit"], "2")
        self.assertEqual(blocked.headers["ratelimit-remaining"], "0")
        self.assertEqual(
            blocked.json(),
            {
                "error": {
                    "code": "rate_limited",
                    "message": "Too many requests. Try again later.",
                }
            },
        )
        self.assertEqual(len(pipeline.calls), 2)

    def test_missing_limiter_fails_readiness_without_leaking_details(self) -> None:
        def fail_to_build_limiter() -> TokenBucketRateLimiter:
            raise RuntimeError("private infrastructure detail")

        with TestClient(
            create_app(
                pipeline=FakePipeline(),
                authenticator=test_authenticator(),
                document_permissions=test_permissions(),
                rate_limiter_factory=fail_to_build_limiter,
            )
        ) as client:
            ready = client.get("/health/ready")
            query = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
                headers=auth_headers(),
            )

        self.assertEqual(ready.status_code, 503)
        self.assertEqual(query.status_code, 503)
        self.assertNotIn("private infrastructure", ready.text + query.text)


if __name__ == "__main__":
    unittest.main()
