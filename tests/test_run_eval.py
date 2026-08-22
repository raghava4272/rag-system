"""Tests for the Phase 3 evaluation CLI and quality gate."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from eval.dataset import GoldenCase
from eval.run_eval import (
    EvaluationConfig,
    evaluate_quality_gate,
    main,
    run_evaluation,
)
from eval.runner import EvaluationCaseResult
from eval.scoring import ScoredCaseResult, ScoringReport
from src.embed import RetrievedChunk
from src.generate import Citation, GeneratedAnswer
from src.pipeline import RAGRun


class FakeMetricResult:
    def __init__(self, value: float) -> None:
        self.value = value


class FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, **_: object) -> FakeMetricResult:
        return FakeMetricResult(self.value)


class FakePipeline:
    def __init__(self, run: RAGRun) -> None:
        self.result = run
        self.calls: list[tuple[str, int]] = []

    def run(self, question: str, *, top_k: int) -> RAGRun:
        self.calls.append((question, top_k))
        return self.result


def scored_report(
    *,
    faithfulness: float = 0.8,
    answer_relevancy: float = 0.7,
    declined: bool = True,
    retrieval_met: bool = True,
) -> ScoringReport:
    supported_case = GoldenCase("Supported?", "Expected.", ("source.mdx",))
    supported_evaluation = EvaluationCaseResult(
        case=supported_case,
        actual_answer="Actual [1].",
        contexts=("Context.",),
        retrieved_sources=("source.mdx",),
        cited_sources=("source.mdx",),
        declined=False,
        expected_source_retrieved=retrieval_met,
        retrieval_expectation_met=retrieval_met,
    )
    decline_case = GoldenCase(
        "Unknown?",
        "Decline.",
        (),
        should_decline=True,
    )
    decline_evaluation = EvaluationCaseResult(
        case=decline_case,
        actual_answer="No context.",
        contexts=(),
        retrieved_sources=(),
        cited_sources=(),
        declined=declined,
        expected_source_retrieved=False,
        retrieval_expectation_met=declined,
    )
    return ScoringReport(
        results=[
            ScoredCaseResult(
                supported_evaluation,
                faithfulness,
                answer_relevancy,
            ),
            ScoredCaseResult(decline_evaluation, None, None),
        ]
    )


class QualityGateTests(unittest.TestCase):
    def test_gate_passes_at_threshold_when_retrieval_and_declines_are_correct(self) -> None:
        gate = evaluate_quality_gate(scored_report(faithfulness=0.8), threshold=0.8)

        self.assertTrue(gate.faithfulness_passed)
        self.assertTrue(gate.declines_passed)
        self.assertTrue(gate.retrieval_passed)
        self.assertTrue(gate.passed)

    def test_each_required_quality_condition_can_fail_the_gate(self) -> None:
        low_faithfulness = evaluate_quality_gate(
            scored_report(faithfulness=0.79),
            threshold=0.8,
        )
        failed_decline = evaluate_quality_gate(
            scored_report(declined=False),
            threshold=0.8,
        )
        failed_retrieval = evaluate_quality_gate(
            scored_report(retrieval_met=False),
            threshold=0.8,
        )

        self.assertFalse(low_faithfulness.passed)
        self.assertFalse(failed_decline.passed)
        self.assertFalse(failed_retrieval.passed)

    def test_threshold_outside_zero_to_one_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            evaluate_quality_gate(scored_report(), threshold=-0.1)
        with self.assertRaisesRegex(ValueError, "threshold"):
            evaluate_quality_gate(scored_report(), threshold=1.1)


class EvaluationOrchestrationTests(unittest.TestCase):
    def test_loads_runs_and_scores_the_configured_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "golden.jsonl"
            dataset.write_text(
                '{"question":"What is RAG?",'
                '"ground_truth_answer":"Retrieval-augmented generation.",'
                '"ground_truth_sources":["source.mdx"]}\n',
                encoding="utf-8",
            )
            retrieved = RetrievedChunk(
                id="chunk",
                text="RAG uses retrieved context.",
                metadata={"source": "source.mdx", "chunk_index": 0},
                distance=0.1,
                rank=1,
            )
            pipeline = FakePipeline(
                RAGRun(
                    GeneratedAnswer(
                        "RAG uses retrieved context [1].",
                        [Citation(1, "chunk", "source.mdx", None, 0)],
                    ),
                    (retrieved,),
                )
            )
            pipeline_configs: list[EvaluationConfig] = []
            scorer_configs: list[EvaluationConfig] = []

            def pipeline_factory(config: EvaluationConfig) -> FakePipeline:
                pipeline_configs.append(config)
                return pipeline

            def scorer_factory(
                config: EvaluationConfig,
            ) -> tuple[FakeMetric, FakeMetric]:
                scorer_configs.append(config)
                return FakeMetric(0.9), FakeMetric(0.8)

            config = EvaluationConfig(
                dataset=dataset,
                threshold=0.8,
                top_k=3,
                min_cases=1,
            )
            report = run_evaluation(
                config,
                pipeline_factory=pipeline_factory,
                scorer_factory=scorer_factory,
            )

        self.assertEqual(pipeline.calls, [("What is RAG?", 3)])
        self.assertEqual(pipeline_configs, [config])
        self.assertEqual(scorer_configs, [config])
        self.assertEqual(report.average_faithfulness, 0.9)
        self.assertEqual(report.average_answer_relevancy, 0.8)


class EvaluationCliTests(unittest.TestCase):
    def test_intentional_failure_drill_exits_one_and_explains_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "intentional-failure.md"
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = main(
                    ["--threshold", "0.8", "--report", str(report_path)],
                    evaluation_function=lambda _: scored_report(
                        faithfulness=0.25,
                        retrieval_met=False,
                    ),
                )

            markdown = report_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 1)
        self.assertIn("Quality gate: FAIL", output.getvalue())
        self.assertIn("**Overall status:** FAIL", markdown)
        self.assertIn("Faithfulness 25.00% is below 80.00%", markdown)
        self.assertIn("Expected source was not retrieved", markdown)

    def test_cli_prints_summary_and_returns_gate_status(self) -> None:
        calls: list[EvaluationConfig] = []

        def passing(config: EvaluationConfig) -> ScoringReport:
            calls.append(config)
            return scored_report()

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "passing.md"
            failed_report_path = Path(directory) / "failing.md"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "--dataset",
                        "custom.jsonl",
                        "--threshold",
                        "0.8",
                        "--top-k",
                        "4",
                        "--report",
                        str(report_path),
                    ],
                    evaluation_function=passing,
                )

            with contextlib.redirect_stdout(io.StringIO()):
                failed_code = main(
                    ["--report", str(failed_report_path)],
                    evaluation_function=lambda _: scored_report(faithfulness=0.5),
                )

            self.assertTrue(report_path.is_file())
            self.assertTrue(failed_report_path.is_file())

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0].dataset, Path("custom.jsonl"))
        self.assertEqual(calls[0].top_k, 4)
        self.assertEqual(calls[0].report_path, report_path)
        self.assertIn("Average faithfulness: 80.00%", output.getvalue())
        self.assertIn("Quality gate: PASS", output.getvalue())
        self.assertIn("Report:", output.getvalue())
        self.assertEqual(failed_code, 1)

    def test_cli_rejects_invalid_threshold_before_evaluation(self) -> None:
        calls: list[EvaluationConfig] = []

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(
                    ["--threshold", "1.1"],
                    evaluation_function=lambda config: calls.append(config),
                )

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
