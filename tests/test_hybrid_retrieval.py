"""Behavioral tests for reciprocal-rank hybrid retrieval."""

from __future__ import annotations

import unittest

from src.embed import RetrievedChunk
from src.hybrid_retrieve import HybridRetriever, reciprocal_rank_fusion
from src.retrieve_bm25 import BM25Result


def vector_result(
    record_id: str,
    *,
    rank: int,
    distance: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=record_id,
        text=f"Evidence for {record_id}",
        metadata={"source": f"{record_id}.md", "chunk_index": 0},
        distance=distance,
        rank=rank,
    )


def bm25_result(
    record_id: str,
    *,
    rank: int,
    score: float,
) -> BM25Result:
    return BM25Result(
        id=record_id,
        text=f"Evidence for {record_id}",
        metadata={"source": f"{record_id}.md", "chunk_index": 0},
        score=score,
        rank=rank,
    )


class FakeRetriever:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, frozenset[str] | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[object]:
        self.calls.append((query, top_k, allowed_sources))
        return self.results


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_chunk_found_by_both_retrievers_is_promoted(self) -> None:
        vector = [
            vector_result("semantic-only", rank=1, distance=0.05),
            vector_result("shared", rank=2, distance=0.20),
        ]
        keyword = [
            bm25_result("shared", rank=1, score=12.0),
            bm25_result("keyword-only", rank=2, score=8.0),
        ]

        results = reciprocal_rank_fusion(vector, keyword, top_k=3, rrf_k=60)

        self.assertEqual([result.id for result in results], [
            "shared",
            "semantic-only",
            "keyword-only",
        ])
        self.assertAlmostEqual(results[0].rrf_score, (1 / 62) + (1 / 61))
        self.assertEqual(results[0].retrieved_by, ("vector", "bm25"))

    def test_fusion_uses_rank_instead_of_incompatible_raw_scores(self) -> None:
        first = reciprocal_rank_fusion(
            [vector_result("vector", rank=1, distance=0.01)],
            [bm25_result("keyword", rank=1, score=1_000.0)],
            top_k=2,
        )
        second = reciprocal_rank_fusion(
            [vector_result("vector", rank=1, distance=999.0)],
            [bm25_result("keyword", rank=1, score=0.001)],
            top_k=2,
        )

        self.assertEqual(first, second)

    def test_unique_results_keep_text_metadata_and_source_method(self) -> None:
        results = reciprocal_rank_fusion(
            [vector_result("semantic-only", rank=1, distance=0.1)],
            [bm25_result("keyword-only", rank=1, score=4.0)],
            top_k=2,
        )

        by_id = {result.id: result for result in results}
        self.assertEqual(
            by_id["semantic-only"].metadata["source"],
            "semantic-only.md",
        )
        self.assertEqual(by_id["semantic-only"].retrieved_by, ("vector",))
        self.assertEqual(by_id["keyword-only"].retrieved_by, ("bm25",))

    def test_top_k_limits_results_and_assigns_consecutive_ranks(self) -> None:
        results = reciprocal_rank_fusion(
            [
                vector_result("a", rank=1, distance=0.1),
                vector_result("b", rank=2, distance=0.2),
            ],
            [bm25_result("c", rank=1, score=3.0)],
            top_k=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_empty_rankings_produce_no_results(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([], [], top_k=5), [])


class HybridRetrieverTests(unittest.TestCase):
    def test_both_retrievers_receive_larger_candidate_limit_before_fusion(self) -> None:
        vector = FakeRetriever([vector_result("shared", rank=1, distance=0.1)])
        keyword = FakeRetriever([bm25_result("shared", rank=1, score=5.0)])
        retriever = HybridRetriever(
            vector_retriever=vector,
            bm25_retriever=keyword,
            candidate_k=10,
        )

        results = retriever.retrieve("How does retrieval work?", top_k=3)

        self.assertEqual(vector.calls, [("How does retrieval work?", 10, None)])
        self.assertEqual(keyword.calls, [("How does retrieval work?", 10, None)])
        self.assertEqual([result.id for result in results], ["shared"])

    def test_permission_scope_is_forwarded_to_both_retrieval_methods(self) -> None:
        vector = FakeRetriever([])
        keyword = FakeRetriever([])
        retriever = HybridRetriever(
            vector_retriever=vector,
            bm25_retriever=keyword,
            candidate_k=10,
        )
        allowed = frozenset({"team-a.md"})

        retriever.retrieve("private question", top_k=3, allowed_sources=allowed)

        self.assertEqual(vector.calls, [("private question", 10, allowed)])
        self.assertEqual(keyword.calls, [("private question", 10, allowed)])

    def test_invalid_query_and_limits_are_rejected_before_search(self) -> None:
        vector = FakeRetriever([])
        keyword = FakeRetriever([])
        retriever = HybridRetriever(
            vector_retriever=vector,
            bm25_retriever=keyword,
            candidate_k=10,
        )

        with self.assertRaisesRegex(ValueError, "query"):
            retriever.retrieve("  \n", top_k=3)
        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.retrieve("question", top_k=0)

        self.assertEqual(vector.calls, [])
        self.assertEqual(keyword.calls, [])


if __name__ == "__main__":
    unittest.main()
