"""Run golden cases through the RAG pipeline and collect scorer inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from eval.dataset import GoldenCase
from src.pipeline import RAGRun


class EvaluationPipeline(Protocol):
    def run(self, question: str, *, top_k: int) -> RAGRun: ...


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    """Observed pipeline behavior for one golden case."""

    case: GoldenCase
    actual_answer: str
    contexts: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    cited_sources: tuple[str, ...]
    declined: bool
    expected_source_retrieved: bool
    retrieval_expectation_met: bool


def _retrieved_source(run: RAGRun, *, index: int) -> str:
    source = run.retrieved_chunks[index].metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            f"retrieved chunk {index + 1} has missing or invalid source metadata"
        )
    return source.strip()


def _source_matches(source: str, expected_sources: Sequence[str]) -> bool:
    return any(expected_source in source for expected_source in expected_sources)


def evaluate_cases(
    cases: Sequence[GoldenCase],
    *,
    pipeline: EvaluationPipeline,
    top_k: int = 5,
) -> list[EvaluationCaseResult]:
    """Execute cases in order and preserve all inputs needed by later scorers."""

    if not cases:
        raise ValueError("cases cannot be empty")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    results: list[EvaluationCaseResult] = []
    for case in cases:
        run = pipeline.run(case.question, top_k=top_k)
        contexts = tuple(chunk.text for chunk in run.retrieved_chunks)
        retrieved_sources = tuple(
            _retrieved_source(run, index=index)
            for index in range(len(run.retrieved_chunks))
        )
        cited_sources = tuple(
            citation.source for citation in run.answer.citations
        )
        declined = not run.retrieved_chunks
        expected_source_retrieved = any(
            _source_matches(source, case.ground_truth_sources)
            for source in retrieved_sources
        )
        retrieval_expectation_met = (
            declined if case.should_decline else expected_source_retrieved
        )
        results.append(
            EvaluationCaseResult(
                case=case,
                actual_answer=run.answer.text,
                contexts=contexts,
                retrieved_sources=retrieved_sources,
                cited_sources=cited_sources,
                declined=declined,
                expected_source_retrieved=expected_source_retrieved,
                retrieval_expectation_met=retrieval_expectation_met,
            )
        )
    return results
