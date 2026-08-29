"""Behavioral tests for reranking, relevance filtering, and safe decline."""

from __future__ import annotations

import unittest

from src.generate import NO_CONTEXT_ANSWER
from src.hybrid_retrieve import HybridResult
from src.pipeline import RAGPipeline
from src.rerank import RerankedResult, RerankingRetriever


def hybrid_candidate(record_id: str, *, rank: int) -> HybridResult:
    return HybridResult(
        id=record_id,
        text=f"Evidence for {record_id}",
        metadata={"source": f"{record_id}.md", "chunk_index": 0},
        rrf_score=1 / (60 + rank),
        rank=rank,
        retrieved_by=("vector", "bm25"),
    )


def reranked_result(
    record_id: str,
    *,
    rank: int,
    relevance_score: float,
) -> RerankedResult:
    return RerankedResult(
        id=record_id,
        text=f"Evidence for {record_id}",
        metadata={"source": f"{record_id}.md", "chunk_index": 0},
        relevance_score=relevance_score,
        rank=rank,
        rrf_score=1 / (60 + rank),
        retrieved_by=("vector", "bm25"),
    )


class FakeHybridRetriever:
    def __init__(self, results: list[HybridResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, frozenset[str] | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[HybridResult]:
        self.calls.append((query, top_k, allowed_sources))
        return self.results


class FakeReranker:
    def __init__(self, results: list[RerankedResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[HybridResult], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        *,
        top_k: int,
    ) -> list[RerankedResult]:
        self.calls.append((query, list(candidates), top_k))
        return self.results


class FakeLanguageModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "This should not be called."


class RerankingRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            hybrid_candidate("first", rank=1),
            hybrid_candidate("second", rank=2),
            hybrid_candidate("third", rank=3),
        ]

    def test_hybrid_candidates_are_reranked_before_final_limit(self) -> None:
        hybrid = FakeHybridRetriever(self.candidates)
        expected = [reranked_result("second", rank=1, relevance_score=0.9)]
        reranker = FakeReranker(expected)
        retriever = RerankingRetriever(
            hybrid_retriever=hybrid,
            reranker=reranker,
            candidate_k=10,
            min_relevance_score=0.5,
        )

        results = retriever.retrieve("question", top_k=2)

        self.assertEqual(hybrid.calls, [("question", 10, None)])
        self.assertEqual(reranker.calls, [("question", self.candidates, 2)])
        self.assertEqual(results, expected)

    def test_permission_scope_reaches_hybrid_retrieval_before_reranking(self) -> None:
        hybrid = FakeHybridRetriever([])
        reranker = FakeReranker([])
        retriever = RerankingRetriever(
            hybrid_retriever=hybrid,
            reranker=reranker,
        )
        allowed = frozenset({"team-a.md"})

        self.assertEqual(
            retriever.retrieve("private question", top_k=3, allowed_sources=allowed),
            [],
        )
        self.assertEqual(hybrid.calls, [("private question", 10, allowed)])
        self.assertEqual(reranker.calls, [])

    def test_chunks_below_relevance_threshold_are_removed(self) -> None:
        hybrid = FakeHybridRetriever(self.candidates)
        reranker = FakeReranker([
            reranked_result("first", rank=1, relevance_score=0.91),
            reranked_result("second", rank=2, relevance_score=0.49),
            reranked_result("third", rank=3, relevance_score=0.10),
        ])
        retriever = RerankingRetriever(
            hybrid_retriever=hybrid,
            reranker=reranker,
            min_relevance_score=0.5,
        )

        results = retriever.retrieve("question", top_k=3)

        self.assertEqual([result.id for result in results], ["first"])

    def test_all_weak_candidates_produce_no_context(self) -> None:
        retriever = RerankingRetriever(
            hybrid_retriever=FakeHybridRetriever(self.candidates),
            reranker=FakeReranker([
                reranked_result("first", rank=1, relevance_score=0.30),
                reranked_result("second", rank=2, relevance_score=0.05),
            ]),
            min_relevance_score=0.5,
        )

        self.assertEqual(retriever.retrieve("unsupported question", top_k=3), [])

    def test_empty_hybrid_candidates_skip_the_reranker(self) -> None:
        hybrid = FakeHybridRetriever([])
        reranker = FakeReranker([])
        retriever = RerankingRetriever(
            hybrid_retriever=hybrid,
            reranker=reranker,
        )

        self.assertEqual(retriever.retrieve("question", top_k=3), [])
        self.assertEqual(reranker.calls, [])

    def test_invalid_inputs_are_rejected_before_retrieval(self) -> None:
        hybrid = FakeHybridRetriever(self.candidates)
        reranker = FakeReranker([])
        retriever = RerankingRetriever(
            hybrid_retriever=hybrid,
            reranker=reranker,
        )

        with self.assertRaisesRegex(ValueError, "query"):
            retriever.retrieve("  \n", top_k=3)
        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.retrieve("question", top_k=0)

        self.assertEqual(hybrid.calls, [])
        self.assertEqual(reranker.calls, [])

    def test_invalid_configuration_is_rejected(self) -> None:
        hybrid = FakeHybridRetriever([])
        reranker = FakeReranker([])

        for threshold in (-0.01, 1.01):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "min_relevance_score"):
                    RerankingRetriever(
                        hybrid_retriever=hybrid,
                        reranker=reranker,
                        min_relevance_score=threshold,
                    )

        with self.assertRaisesRegex(ValueError, "candidate_k"):
            RerankingRetriever(
                hybrid_retriever=hybrid,
                reranker=reranker,
                candidate_k=0,
            )

    def test_low_relevance_pipeline_declines_without_calling_groq(self) -> None:
        retriever = RerankingRetriever(
            hybrid_retriever=FakeHybridRetriever(self.candidates),
            reranker=FakeReranker([
                reranked_result("first", rank=1, relevance_score=0.20),
            ]),
            min_relevance_score=0.5,
        )
        language_model = FakeLanguageModel()
        pipeline = RAGPipeline(
            retriever=retriever,
            language_model=language_model,
        )

        answer = pipeline.ask("What is the capital of Mars?", top_k=3)

        self.assertEqual(answer.text, NO_CONTEXT_ANSWER)
        self.assertEqual(answer.citations, [])
        self.assertEqual(language_model.calls, [])


if __name__ == "__main__":
    unittest.main()
