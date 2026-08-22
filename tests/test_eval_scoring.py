"""Behavioral tests for Phase 3 metric scoring and aggregation."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from eval.dataset import GoldenCase
from eval.runner import EvaluationCaseResult
from eval.scoring import score_evaluation_results


@dataclass(frozen=True)
class FakeMetricResult:
    value: float


class FakeMetric:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.calls: list[dict[str, object]] = []

    def score(self, **kwargs: object) -> FakeMetricResult:
        self.calls.append(kwargs)
        return FakeMetricResult(next(self.values))


def supported_result(
    question: str,
    *,
    answer: str = "A grounded answer [1].",
    contexts: tuple[str, ...] = ("Grounding context.",),
    source_found: bool = True,
) -> EvaluationCaseResult:
    case = GoldenCase(
        question=question,
        ground_truth_answer="Expected answer.",
        ground_truth_sources=("source.mdx",),
    )
    return EvaluationCaseResult(
        case=case,
        actual_answer=answer,
        contexts=contexts,
        retrieved_sources=("source.mdx",) if source_found else ("other.mdx",),
        cited_sources=("source.mdx",),
        declined=False,
        expected_source_retrieved=source_found,
        retrieval_expectation_met=source_found,
    )


def decline_result(question: str, *, declined: bool) -> EvaluationCaseResult:
    case = GoldenCase(
        question=question,
        ground_truth_answer="The corpus cannot answer this.",
        ground_truth_sources=(),
        should_decline=True,
    )
    return EvaluationCaseResult(
        case=case,
        actual_answer="I could not find relevant source material.",
        contexts=() if declined else ("Weak unrelated context.",),
        retrieved_sources=() if declined else ("other.mdx",),
        cited_sources=(),
        declined=declined,
        expected_source_retrieved=False,
        retrieval_expectation_met=declined,
    )


class EvaluationScoringTests(unittest.TestCase):
    def test_supported_case_is_scored_with_the_correct_metric_inputs(self) -> None:
        result = supported_result("What is RAG?")
        faithfulness = FakeMetric([0.9])
        relevancy = FakeMetric([0.8])

        report = score_evaluation_results(
            [result],
            faithfulness_scorer=faithfulness,
            answer_relevancy_scorer=relevancy,
        )

        self.assertEqual(
            faithfulness.calls,
            [
                {
                    "user_input": result.case.question,
                    "response": result.actual_answer,
                    "retrieved_contexts": list(result.contexts),
                }
            ],
        )
        self.assertEqual(
            relevancy.calls,
            [
                {
                    "user_input": result.case.question,
                    "response": result.actual_answer,
                }
            ],
        )
        self.assertEqual(report.results[0].faithfulness, 0.9)
        self.assertEqual(report.results[0].answer_relevancy, 0.8)

    def test_decline_cases_skip_llm_metrics_and_use_behavioral_scoring(self) -> None:
        correct = decline_result("Unknown one?", declined=True)
        incorrect = decline_result("Unknown two?", declined=False)
        faithfulness = FakeMetric([])
        relevancy = FakeMetric([])

        report = score_evaluation_results(
            [correct, incorrect],
            faithfulness_scorer=faithfulness,
            answer_relevancy_scorer=relevancy,
        )

        self.assertEqual(faithfulness.calls, [])
        self.assertEqual(relevancy.calls, [])
        self.assertIsNone(report.results[0].faithfulness)
        self.assertIsNone(report.results[0].answer_relevancy)
        self.assertEqual(report.decline_accuracy, 0.5)

    def test_report_averages_only_supported_cases(self) -> None:
        results = [
            supported_result("First?"),
            decline_result("Unknown?", declined=True),
            supported_result("Second?"),
        ]

        report = score_evaluation_results(
            results,
            faithfulness_scorer=FakeMetric([0.9, 0.7]),
            answer_relevancy_scorer=FakeMetric([0.8, 0.6]),
        )

        self.assertAlmostEqual(report.average_faithfulness, 0.8)
        self.assertAlmostEqual(report.average_answer_relevancy, 0.7)
        self.assertEqual(report.decline_accuracy, 1.0)
        self.assertEqual(report.retrieval_expectation_rate, 1.0)

    def test_invalid_or_non_finite_metric_values_are_rejected(self) -> None:
        result = supported_result("What is RAG?")

        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "faithfulness"):
                    score_evaluation_results(
                        [result],
                        faithfulness_scorer=FakeMetric([invalid]),
                        answer_relevancy_scorer=FakeMetric([0.8]),
                    )

        for invalid in (-1.1, 1.1, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "answer_relevancy"):
                    score_evaluation_results(
                        [result],
                        faithfulness_scorer=FakeMetric([0.8]),
                        answer_relevancy_scorer=FakeMetric([invalid]),
                    )

    def test_empty_results_are_rejected_before_metric_calls(self) -> None:
        faithfulness = FakeMetric([])
        relevancy = FakeMetric([])

        with self.assertRaisesRegex(ValueError, "results"):
            score_evaluation_results(
                [],
                faithfulness_scorer=faithfulness,
                answer_relevancy_scorer=relevancy,
            )

        self.assertEqual(faithfulness.calls, [])
        self.assertEqual(relevancy.calls, [])


if __name__ == "__main__":
    unittest.main()
