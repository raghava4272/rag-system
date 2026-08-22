"""Metrics and pass/fail rules for the Phase 2 retrieval regression gate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class SourceResult(Protocol):
    metadata: dict[str, str | int | float | bool]


class SourceRetriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> Sequence[SourceResult]: ...


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    question: str
    expected_source_fragment: str


@dataclass(frozen=True, slots=True)
class SupportedCaseResult:
    case: RetrievalCase
    baseline_rank: int | None
    upgraded_rank: int | None


@dataclass(frozen=True, slots=True)
class UnsupportedCaseResult:
    question: str
    declined: bool


@dataclass(frozen=True, slots=True)
class RegressionReport:
    supported_results: list[SupportedCaseResult]
    unsupported_results: list[UnsupportedCaseResult]

    @staticmethod
    def _hit_rate(
        results: Sequence[SupportedCaseResult],
        *,
        system: str,
        at_k: int,
    ) -> float:
        ranks = [getattr(result, f"{system}_rank") for result in results]
        hits = sum(rank is not None and rank <= at_k for rank in ranks)
        return hits / len(ranks)

    @property
    def baseline_hit_at_1(self) -> float:
        return self._hit_rate(self.supported_results, system="baseline", at_k=1)

    @property
    def upgraded_hit_at_1(self) -> float:
        return self._hit_rate(self.supported_results, system="upgraded", at_k=1)

    @property
    def baseline_hit_at_3(self) -> float:
        return self._hit_rate(self.supported_results, system="baseline", at_k=3)

    @property
    def upgraded_hit_at_3(self) -> float:
        return self._hit_rate(self.supported_results, system="upgraded", at_k=3)

    @property
    def unsupported_decline_rate(self) -> float:
        declines = sum(result.declined for result in self.unsupported_results)
        return declines / len(self.unsupported_results)

    @staticmethod
    def _comparable_rank(rank: int | None) -> float:
        return float("inf") if rank is None else float(rank)

    @property
    def no_retrieval_regression(self) -> bool:
        return all(
            self._comparable_rank(result.upgraded_rank)
            <= self._comparable_rank(result.baseline_rank)
            for result in self.supported_results
        )

    @property
    def measurably_improved(self) -> bool:
        return any(
            self._comparable_rank(result.upgraded_rank)
            < self._comparable_rank(result.baseline_rank)
            for result in self.supported_results
        )

    @property
    def phase2_ready(self) -> bool:
        return (
            self.no_retrieval_regression
            and self.measurably_improved
            and self.unsupported_decline_rate == 1.0
        )


def _source_rank(
    results: Sequence[SourceResult],
    expected_source_fragment: str,
) -> int | None:
    for rank, result in enumerate(results, start=1):
        source = result.metadata.get("source")
        if isinstance(source, str) and expected_source_fragment in source:
            return rank
    return None


def evaluate_phase2(
    *,
    baseline_retriever: SourceRetriever,
    upgraded_retriever: SourceRetriever,
    supported_cases: Sequence[RetrievalCase],
    unsupported_questions: Sequence[str],
    top_k: int = 3,
) -> RegressionReport:
    """Compare source ranks and verify that unsupported questions are declined."""

    if not supported_cases:
        raise ValueError("supported_cases cannot be empty")
    if not unsupported_questions:
        raise ValueError("unsupported_questions cannot be empty")
    if top_k < 3:
        raise ValueError("top_k must be at least three to measure Hit@3")
    if any(not case.question.strip() for case in supported_cases):
        raise ValueError("supported case questions cannot be empty")
    if any(not case.expected_source_fragment.strip() for case in supported_cases):
        raise ValueError("expected source fragments cannot be empty")
    if any(not question.strip() for question in unsupported_questions):
        raise ValueError("unsupported questions cannot be empty")

    supported_results: list[SupportedCaseResult] = []
    for case in supported_cases:
        baseline_results = baseline_retriever.retrieve(case.question, top_k=top_k)
        upgraded_results = upgraded_retriever.retrieve(case.question, top_k=top_k)
        supported_results.append(
            SupportedCaseResult(
                case=case,
                baseline_rank=_source_rank(
                    baseline_results,
                    case.expected_source_fragment,
                ),
                upgraded_rank=_source_rank(
                    upgraded_results,
                    case.expected_source_fragment,
                ),
            )
        )

    unsupported_results = [
        UnsupportedCaseResult(
            question=question,
            declined=not upgraded_retriever.retrieve(question, top_k=top_k),
        )
        for question in unsupported_questions
    ]
    return RegressionReport(
        supported_results=supported_results,
        unsupported_results=unsupported_results,
    )
