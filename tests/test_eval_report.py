"""Tests for the human-readable Phase 3 Markdown artifact."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from eval.dataset import GoldenCase
from eval.report import render_markdown_report, write_markdown_report
from eval.runner import EvaluationCaseResult
from eval.scoring import ScoredCaseResult, ScoringReport


@dataclass(frozen=True)
class FakeGate:
    report: ScoringReport
    threshold: float
    faithfulness_passed: bool
    declines_passed: bool
    retrieval_passed: bool
    passed: bool


def supported(
    *,
    question: str = "What is RAG?",
    faithfulness: float = 0.9,
    retrieval_met: bool = True,
) -> ScoredCaseResult:
    case = GoldenCase(
        question,
        "RAG combines retrieval and generation.",
        ("docs/rag|guide.mdx",),
    )
    evaluation = EvaluationCaseResult(
        case=case,
        actual_answer="RAG uses retrieved evidence [1].",
        contexts=("Retrieved evidence.",),
        retrieved_sources=(
            ("docs/rag|guide.mdx",) if retrieval_met else ("docs/other.mdx",)
        ),
        cited_sources=("docs/rag|guide.mdx",),
        declined=False,
        expected_source_retrieved=retrieval_met,
        retrieval_expectation_met=retrieval_met,
    )
    return ScoredCaseResult(evaluation, faithfulness, 0.8)


def decline(*, declined: bool = True) -> ScoredCaseResult:
    case = GoldenCase(
        "What is Mars' capital?",
        "The corpus cannot answer this.",
        (),
        should_decline=True,
    )
    evaluation = EvaluationCaseResult(
        case=case,
        actual_answer=(
            "I could not find relevant source material."
            if declined
            else "Mars has a capital [1]."
        ),
        contexts=() if declined else ("Unrelated context.",),
        retrieved_sources=() if declined else ("docs/other.mdx",),
        cited_sources=() if declined else ("docs/other.mdx",),
        declined=declined,
        expected_source_retrieved=False,
        retrieval_expectation_met=declined,
    )
    return ScoredCaseResult(evaluation, None, None)


class MarkdownReportTests(unittest.TestCase):
    def test_passing_report_contains_aggregates_and_per_case_table(self) -> None:
        report = ScoringReport([supported(), decline()])
        gate = FakeGate(report, 0.8, True, True, True, True)

        markdown = render_markdown_report(gate)

        self.assertIn("# RAG Evaluation Report", markdown)
        self.assertIn("**Overall status:** PASS", markdown)
        self.assertIn("Average faithfulness | 90.00%", markdown)
        self.assertIn("| 1 | PASS | Supported", markdown)
        self.assertIn("| 2 | PASS | Expected decline", markdown)
        self.assertIn("docs/rag\\|guide.mdx", markdown)
        self.assertIn("No failing cases.", markdown)

    def test_failure_details_explain_low_score_missing_source_and_failed_decline(self) -> None:
        report = ScoringReport(
            [
                supported(faithfulness=0.5, retrieval_met=False),
                decline(declined=False),
            ]
        )
        gate = FakeGate(report, 0.8, False, False, False, False)

        markdown = render_markdown_report(gate)

        self.assertIn("**Overall status:** FAIL", markdown)
        self.assertIn("Faithfulness 50.00% is below 80.00%", markdown)
        self.assertIn("Expected source was not retrieved", markdown)
        self.assertIn("Expected decline was not honored", markdown)
        self.assertIn("Expected answer", markdown)
        self.assertIn("Actual answer", markdown)
        self.assertIn("docs/other.mdx", markdown)

    def test_writer_creates_parent_directories_and_exact_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts" / "rag-eval.md"

            write_markdown_report(path, "# Report\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "# Report\n")


if __name__ == "__main__":
    unittest.main()
