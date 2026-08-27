"""HTTP contract tests for the RAG API boundary."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api import create_app
from src.generate import Citation, GeneratedAnswer, NO_CONTEXT_ANSWER


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
        self.calls: list[tuple[str, int]] = []

    def ask(self, question: str, *, top_k: int) -> GeneratedAnswer:
        self.calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        return self.answer


class ApiHealthTests(unittest.TestCase):
    def test_liveness_and_readiness_are_distinct(self) -> None:
        with TestClient(create_app(pipeline=FakePipeline())) as client:
            live_response = client.get("/health/live")
            ready_response = client.get("/health/ready")

        self.assertEqual(live_response.status_code, 200)
        self.assertEqual(live_response.json(), {"status": "alive"})
        self.assertEqual(ready_response.status_code, 200)
        self.assertEqual(ready_response.json(), {"status": "ready"})

    def test_failed_startup_stays_live_but_is_not_ready(self) -> None:
        def fail_to_build() -> FakePipeline:
            raise RuntimeError("GROQ_API_KEY accidentally leaked in this message")

        with TestClient(create_app(pipeline_factory=fail_to_build)) as client:
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

        with TestClient(create_app(pipeline_factory=build_pipeline)) as client:
            client.post("/v1/query", json={"question": "First?"})
            client.post("/v1/query", json={"question": "Second?"})

        self.assertEqual(factory_calls, [None])


class ApiQueryTests(unittest.TestCase):
    def test_query_returns_a_typed_answer_and_structured_citations(self) -> None:
        pipeline = FakePipeline()

        with TestClient(create_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "  How does hybrid retrieval work?  ", "top_k": 3},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            pipeline.calls,
            [("How does hybrid retrieval work?", 3)],
        )
        self.assertEqual(
            response.json(),
            {
                "answer": pipeline.answer.text,
                "abstained": False,
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

    def test_no_context_answer_is_an_explicit_abstention(self) -> None:
        pipeline = FakePipeline(
            GeneratedAnswer(text=NO_CONTEXT_ANSWER, citations=[]),
        )

        with TestClient(create_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is outside the corpus?"},
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

        with TestClient(create_app(pipeline=pipeline)) as client:
            responses = [
                client.post("/v1/query", json=payload)
                for payload in invalid_payloads
            ]

        self.assertTrue(all(response.status_code == 422 for response in responses))
        self.assertEqual(pipeline.calls, [])

    def test_pipeline_failures_return_a_safe_stable_error(self) -> None:
        pipeline = FakePipeline(
            error=RuntimeError("provider response contained a private detail"),
        )

        with TestClient(create_app(pipeline=pipeline)) as client:
            response = client.post(
                "/v1/query",
                json={"question": "What is RAG?"},
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


if __name__ == "__main__":
    unittest.main()
