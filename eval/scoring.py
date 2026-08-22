"""Score supported answers with Ragas-compatible metrics and declines directly."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from eval.runner import EvaluationCaseResult


class MetricResult(Protocol):
    value: float


class MetricScorer(Protocol):
    def score(self, **kwargs: Any) -> MetricResult: ...


@dataclass(frozen=True, slots=True)
class ScoredCaseResult:
    """One collected pipeline result with optional LLM-judge scores."""

    evaluation: EvaluationCaseResult
    faithfulness: float | None
    answer_relevancy: float | None


@dataclass(frozen=True, slots=True)
class ScoringReport:
    """Per-case scores and aggregate Phase 3 quality measurements."""

    results: list[ScoredCaseResult]

    @property
    def supported_results(self) -> list[ScoredCaseResult]:
        return [
            result
            for result in self.results
            if not result.evaluation.case.should_decline
        ]

    @property
    def decline_results(self) -> list[ScoredCaseResult]:
        return [
            result
            for result in self.results
            if result.evaluation.case.should_decline
        ]

    @staticmethod
    def _average(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @property
    def average_faithfulness(self) -> float | None:
        return self._average(
            [
                result.faithfulness
                for result in self.supported_results
                if result.faithfulness is not None
            ]
        )

    @property
    def average_answer_relevancy(self) -> float | None:
        return self._average(
            [
                result.answer_relevancy
                for result in self.supported_results
                if result.answer_relevancy is not None
            ]
        )

    @property
    def decline_accuracy(self) -> float | None:
        declines = self.decline_results
        if not declines:
            return None
        return sum(result.evaluation.declined for result in declines) / len(declines)

    @property
    def retrieval_expectation_rate(self) -> float:
        matches = sum(
            result.evaluation.retrieval_expectation_met for result in self.results
        )
        return matches / len(self.results)


def _metric_value(
    result: MetricResult,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = result.value
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"{name} returned a non-numeric value")
    value = float(raw_value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must return a finite score between {minimum} and {maximum}"
        )
    return value


def score_evaluation_results(
    results: Sequence[EvaluationCaseResult],
    *,
    faithfulness_scorer: MetricScorer,
    answer_relevancy_scorer: MetricScorer,
) -> ScoringReport:
    """Score supported answers and keep decline evaluation deterministic."""

    if not results:
        raise ValueError("results cannot be empty")

    scored_results: list[ScoredCaseResult] = []
    for result in results:
        if result.case.should_decline:
            scored_results.append(
                ScoredCaseResult(
                    evaluation=result,
                    faithfulness=None,
                    answer_relevancy=None,
                )
            )
            continue

        faithfulness = _metric_value(
            faithfulness_scorer.score(
                user_input=result.case.question,
                response=result.actual_answer,
                retrieved_contexts=list(result.contexts),
            ),
            name="faithfulness",
            minimum=0.0,
            maximum=1.0,
        )
        answer_relevancy = _metric_value(
            answer_relevancy_scorer.score(
                user_input=result.case.question,
                response=result.actual_answer,
            ),
            name="answer_relevancy",
            minimum=-1.0,
            maximum=1.0,
        )
        scored_results.append(
            ScoredCaseResult(
                evaluation=result,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
            )
        )

    return ScoringReport(results=scored_results)


def create_ragas_scorers(
    *,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> tuple[MetricScorer, MetricScorer]:
    """Construct the supported Ragas collections metrics behind one adapter."""

    try:
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness
    except ImportError as error:
        raise RuntimeError(
            "Phase 3 scoring requires Ragas. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from error

    return (
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
    )
