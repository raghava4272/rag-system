"""Behavioral tests for cross-encoder candidate reranking."""

from __future__ import annotations

import unittest

from src.hybrid_retrieve import HybridResult
from src.rerank import CrossEncoderReranker


def candidate(record_id: str, *, rank: int, rrf_score: float) -> HybridResult:
    return HybridResult(
        id=record_id,
        text=f"Evidence for {record_id}",
        metadata={"source": f"{record_id}.md", "chunk_index": 0},
        rrf_score=rrf_score,
        rank=rank,
        retrieved_by=("vector", "bm25"),
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        **options: object,
    ) -> list[float]:
        self.calls.append((pairs, options))
        return self.scores


class CrossEncoderRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            candidate("rrf-first", rank=1, rrf_score=0.032),
            candidate("actually-relevant", rank=2, rrf_score=0.030),
            candidate("weak", rank=3, rrf_score=0.020),
        ]

    def test_model_scores_query_chunk_pairs_and_reorders_candidates(self) -> None:
        model = FakeCrossEncoder([0.15, 0.95, 0.30])
        reranker = CrossEncoderReranker(model=model)

        results = reranker.rerank(
            "Which chunk answers the question?",
            self.candidates,
            top_k=3,
        )

        self.assertEqual(
            model.calls,
            [
                (
                    [
                        (
                            "Which chunk answers the question?",
                            "Evidence for rrf-first",
                        ),
                        (
                            "Which chunk answers the question?",
                            "Evidence for actually-relevant",
                        ),
                        (
                            "Which chunk answers the question?",
                            "Evidence for weak",
                        ),
                    ],
                    {"show_progress_bar": False},
                )
            ],
        )
        self.assertEqual(
            [result.id for result in results],
            ["actually-relevant", "weak", "rrf-first"],
        )
        self.assertEqual(
            [result.relevance_score for result in results],
            [0.95, 0.30, 0.15],
        )
        self.assertEqual([result.rank for result in results], [1, 2, 3])

    def test_reranked_results_preserve_fusion_and_citation_data(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.9, 0.2, 0.1]))

        result = reranker.rerank("question", self.candidates, top_k=1)[0]

        self.assertEqual(result.text, self.candidates[0].text)
        self.assertEqual(result.metadata, self.candidates[0].metadata)
        self.assertEqual(result.rrf_score, self.candidates[0].rrf_score)
        self.assertEqual(result.retrieved_by, self.candidates[0].retrieved_by)

    def test_top_k_limits_output_but_all_candidates_are_scored(self) -> None:
        model = FakeCrossEncoder([0.1, 0.8, 0.5])
        reranker = CrossEncoderReranker(model=model)

        results = reranker.rerank("question", self.candidates, top_k=2)

        self.assertEqual(len(model.calls[0][0]), 3)
        self.assertEqual([result.id for result in results], [
            "actually-relevant",
            "weak",
        ])
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_equal_scores_preserve_the_fused_candidate_order(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.5, 0.5, 0.5]))

        results = reranker.rerank("question", self.candidates, top_k=3)

        self.assertEqual(
            [result.id for result in results],
            ["rrf-first", "actually-relevant", "weak"],
        )

    def test_empty_candidates_return_without_calling_the_model(self) -> None:
        model = FakeCrossEncoder([])
        reranker = CrossEncoderReranker(model=model)

        self.assertEqual(reranker.rerank("question", [], top_k=3), [])
        self.assertEqual(model.calls, [])

    def test_score_count_mismatch_is_rejected(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.5]))

        with self.assertRaisesRegex(ValueError, "score"):
            reranker.rerank("question", self.candidates, top_k=3)

    def test_invalid_query_and_top_k_are_rejected_before_model_call(self) -> None:
        model = FakeCrossEncoder([0.5, 0.4, 0.3])
        reranker = CrossEncoderReranker(model=model)

        with self.assertRaisesRegex(ValueError, "query"):
            reranker.rerank("  \n", self.candidates, top_k=3)
        with self.assertRaisesRegex(ValueError, "top_k"):
            reranker.rerank("question", self.candidates, top_k=0)

        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
