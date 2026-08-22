"""Render Phase 3 evaluation results as a reviewer-friendly Markdown artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from eval.scoring import ScoredCaseResult, ScoringReport


class QualityGate(Protocol):
    report: ScoringReport
    threshold: float
    faithfulness_passed: bool
    declines_passed: bool
    retrieval_passed: bool
    passed: bool


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _table_text(value: str) -> str:
    return value.replace("\n", "<br>").replace("|", "\\|")


def _table_sources(sources: tuple[str, ...]) -> str:
    if not sources:
        return "—"
    return "<br>".join(_table_text(source) for source in sources)


def _case_failures(result: ScoredCaseResult, *, threshold: float) -> list[str]:
    evaluation = result.evaluation
    if evaluation.case.should_decline:
        return [] if evaluation.declined else ["Expected decline was not honored"]

    failures: list[str] = []
    if result.faithfulness is None:
        failures.append("Faithfulness was not scored")
    elif result.faithfulness < threshold:
        failures.append(
            f"Faithfulness {_percentage(result.faithfulness)} is below "
            f"{_percentage(threshold)}"
        )
    if not evaluation.expected_source_retrieved:
        failures.append("Expected source was not retrieved")
    return failures


def _blockquote(value: str) -> list[str]:
    lines = value.splitlines() or [""]
    return [f"> {line}" if line else ">" for line in lines]


def render_markdown_report(gate: QualityGate) -> str:
    """Render aggregate metrics, every case, and expanded failure diagnostics."""

    report = gate.report
    lines = [
        "# RAG Evaluation Report",
        "",
        f"**Overall status:** {'PASS' if gate.passed else 'FAIL'}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value | Requirement | Status |",
        "| --- | ---: | ---: | :---: |",
        (
            f"| Average faithfulness | {_percentage(report.average_faithfulness)} | "
            f">= {_percentage(gate.threshold)} | "
            f"{'PASS' if gate.faithfulness_passed else 'FAIL'} |"
        ),
        (
            "| Average answer relevancy | "
            f"{_percentage(report.average_answer_relevancy)} | Informational | — |"
        ),
        (
            f"| Expected decline accuracy | {_percentage(report.decline_accuracy)} | "
            f"100.00% | {'PASS' if gate.declines_passed else 'FAIL'} |"
        ),
        (
            "| Retrieval expectation rate | "
            f"{_percentage(report.retrieval_expectation_rate)} | 100.00% | "
            f"{'PASS' if gate.retrieval_passed else 'FAIL'} |"
        ),
        "",
        "## Per-case results",
        "",
        (
            "| # | Status | Type | Faithfulness | Relevancy | Retrieval | "
            "Expected sources | Question |"
        ),
        "| ---: | :---: | --- | ---: | ---: | :---: | --- | --- |",
    ]

    failing_cases: list[tuple[int, ScoredCaseResult, list[str]]] = []
    for number, result in enumerate(report.results, start=1):
        evaluation = result.evaluation
        failures = _case_failures(result, threshold=gate.threshold)
        if failures:
            failing_cases.append((number, result, failures))
        case_type = (
            "Expected decline" if evaluation.case.should_decline else "Supported"
        )
        retrieval_status = (
            "PASS" if evaluation.retrieval_expectation_met else "FAIL"
        )
        lines.append(
            "| "
            f"{number} | {'FAIL' if failures else 'PASS'} | {case_type} | "
            f"{_percentage(result.faithfulness)} | "
            f"{_percentage(result.answer_relevancy)} | {retrieval_status} | "
            f"{_table_sources(evaluation.case.ground_truth_sources)} | "
            f"{_table_text(evaluation.case.question)} |"
        )

    lines.extend(("", "## Failure details", ""))
    if not failing_cases:
        lines.append("No failing cases.")
    else:
        for number, result, failures in failing_cases:
            evaluation = result.evaluation
            lines.extend(
                (
                    f"### Case {number}: {evaluation.case.question}",
                    "",
                    "**Why it failed:** " + "; ".join(failures),
                    "",
                    "**Expected answer:**",
                    "",
                    *_blockquote(evaluation.case.ground_truth_answer),
                    "",
                    "**Actual answer:**",
                    "",
                    *_blockquote(evaluation.actual_answer),
                    "",
                    "**Expected sources:** "
                    + (", ".join(evaluation.case.ground_truth_sources) or "None"),
                    "",
                    "**Retrieved sources:** "
                    + (", ".join(evaluation.retrieved_sources) or "None"),
                    "",
                    "**Cited sources:** "
                    + (", ".join(evaluation.cited_sources) or "None"),
                    "",
                )
            )

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(path: str | Path, markdown: str) -> None:
    """Write the report, creating its artifact directory when necessary."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
