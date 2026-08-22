"""Behavioral tests for the Phase 2 retrieval regression gate."""

from __future__ import annotations

import unittest

from src.embed import RetrievedChunk
from src.regression import RetrievalCase, evaluate_phase2


def result(record_id: str, source: str, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=record_id,
        text=f"Evidence from {source}",
        metadata={"source": source, "chunk_index": 0},
        distance=0.1 * rank,
        rank=rank,
    )


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[RetrievedChunk]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])[:top_k]


class Phase2RegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.supported = [
            RetrievalCase("splitting question", "splitters/index.mdx"),
            RetrievalCase("chroma question", "vectorstores/chroma.mdx"),
        ]
        self.unsupported = ["capital of Mars", "sourdough recipe"]

    def test_report_measures_improvement_and_complete_decline_rate(self) -> None:
        baseline = FakeRetriever(
            {
                "splitting question": [
                    result("distractor", "other.mdx", rank=1),
                    result("target", "splitters/index.mdx", rank=2),
                ],
                "chroma question": [
                    result("chroma", "vectorstores/chroma.mdx", rank=1),
                ],
            }
        )
        upgraded = FakeRetriever(
            {
                "splitting question": [
                    result("target", "splitters/index.mdx", rank=1),
                ],
                "chroma question": [
                    result("chroma", "vectorstores/chroma.mdx", rank=1),
                ],
                "capital of Mars": [],
                "sourdough recipe": [],
            }
        )

        report = evaluate_phase2(
            baseline_retriever=baseline,
            upgraded_retriever=upgraded,
            supported_cases=self.supported,
            unsupported_questions=self.unsupported,
            top_k=3,
        )

        self.assertEqual(report.baseline_hit_at_1, 0.5)
        self.assertEqual(report.upgraded_hit_at_1, 1.0)
        self.assertEqual(report.baseline_hit_at_3, 1.0)
        self.assertEqual(report.upgraded_hit_at_3, 1.0)
        self.assertEqual(report.unsupported_decline_rate, 1.0)
        self.assertTrue(report.no_retrieval_regression)
        self.assertTrue(report.measurably_improved)
        self.assertTrue(report.phase2_ready)

    def test_source_rank_is_recorded_for_each_supported_case(self) -> None:
        baseline = FakeRetriever(
            {
                "splitting question": [
                    result("other", "other.mdx", rank=1),
                    result("target", "splitters/index.mdx", rank=2),
                ],
                "chroma question": [],
            }
        )
        upgraded = FakeRetriever(
            {
                "splitting question": [
                    result("target", "splitters/index.mdx", rank=1),
                ],
                "chroma question": [
                    result("chroma", "vectorstores/chroma.mdx", rank=1),
                ],
            }
        )

        report = evaluate_phase2(
            baseline_retriever=baseline,
            upgraded_retriever=upgraded,
            supported_cases=self.supported,
            unsupported_questions=self.unsupported,
            top_k=3,
        )

        self.assertEqual(
            [
                (case.baseline_rank, case.upgraded_rank)
                for case in report.supported_results
            ],
            [(2, 1), (None, 1)],
        )

    def test_any_ranking_regression_or_failed_decline_blocks_readiness(self) -> None:
        baseline = FakeRetriever(
            {
                "splitting question": [
                    result("target", "splitters/index.mdx", rank=1),
                ],
                "chroma question": [
                    result("chroma", "vectorstores/chroma.mdx", rank=1),
                ],
            }
        )
        upgraded = FakeRetriever(
            {
                "splitting question": [],
                "chroma question": [
                    result("chroma", "vectorstores/chroma.mdx", rank=1),
                ],
                "capital of Mars": [result("wrong", "other.mdx", rank=1)],
                "sourdough recipe": [],
            }
        )

        report = evaluate_phase2(
            baseline_retriever=baseline,
            upgraded_retriever=upgraded,
            supported_cases=self.supported,
            unsupported_questions=self.unsupported,
            top_k=3,
        )

        self.assertFalse(report.no_retrieval_regression)
        self.assertEqual(report.unsupported_decline_rate, 0.5)
        self.assertFalse(report.phase2_ready)

    def test_invalid_evaluation_inputs_are_rejected_before_retrieval(self) -> None:
        baseline = FakeRetriever({})
        upgraded = FakeRetriever({})

        with self.assertRaisesRegex(ValueError, "supported_cases"):
            evaluate_phase2(
                baseline_retriever=baseline,
                upgraded_retriever=upgraded,
                supported_cases=[],
                unsupported_questions=self.unsupported,
            )
        with self.assertRaisesRegex(ValueError, "unsupported_questions"):
            evaluate_phase2(
                baseline_retriever=baseline,
                upgraded_retriever=upgraded,
                supported_cases=self.supported,
                unsupported_questions=[],
            )
        with self.assertRaisesRegex(ValueError, "top_k"):
            evaluate_phase2(
                baseline_retriever=baseline,
                upgraded_retriever=upgraded,
                supported_cases=self.supported,
                unsupported_questions=self.unsupported,
                top_k=0,
            )

        self.assertEqual(baseline.calls, [])
        self.assertEqual(upgraded.calls, [])


if __name__ == "__main__":
    unittest.main()
