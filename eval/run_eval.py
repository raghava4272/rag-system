#!/usr/bin/env python3
"""Run the Phase 3 golden evaluation and enforce its quality gate."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.dataset import load_golden_dataset
from eval.report import render_markdown_report, write_markdown_report
from eval.runner import EvaluationPipeline, evaluate_cases
from eval.scoring import (
    MetricScorer,
    ScoringReport,
    create_ragas_scorers,
    score_evaluation_results,
)
from scripts.ask import build_phase2_retriever
from src.embed import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
    SentenceTransformerEmbeddingProvider,
)
from src.llm import DEFAULT_GROQ_MODEL, GROQ_BASE_URL, GroqLanguageModel
from src.pipeline import RAGPipeline
from src.rerank import (
    DEFAULT_MIN_RELEVANCE_SCORE,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
)
from src.retrieve import DEFAULT_TOP_K


DEFAULT_DATASET_PATH = Path("eval/golden_dataset.jsonl")
DEFAULT_REPORT_PATH = Path("eval/report.md")
DEFAULT_FAITHFULNESS_THRESHOLD = 0.8
MINIMUM_GOLDEN_CASES = 50


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    dataset: Path = DEFAULT_DATASET_PATH
    threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD
    top_k: int = DEFAULT_TOP_K
    min_cases: int = MINIMUM_GOLDEN_CASES
    report_path: Path = DEFAULT_REPORT_PATH
    candidate_k: int = DEFAULT_RERANK_CANDIDATE_K
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE
    chroma_path: Path = DEFAULT_CHROMA_PATH
    collection: str = DEFAULT_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    groq_model: str = DEFAULT_GROQ_MODEL
    evaluator_model: str = DEFAULT_GROQ_MODEL


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    report: ScoringReport
    threshold: float
    faithfulness_passed: bool
    declines_passed: bool
    retrieval_passed: bool

    @property
    def passed(self) -> bool:
        return (
            self.faithfulness_passed
            and self.declines_passed
            and self.retrieval_passed
        )


class PipelineFactory(Protocol):
    def __call__(self, config: EvaluationConfig) -> EvaluationPipeline: ...


class ScorerFactory(Protocol):
    def __call__(
        self,
        config: EvaluationConfig,
    ) -> tuple[MetricScorer, MetricScorer]: ...


def _unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the full RAG pipeline against the golden dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report artifact path (default: eval/report.md)",
    )
    parser.add_argument(
        "--threshold",
        type=_unit_interval,
        default=DEFAULT_FAITHFULNESS_THRESHOLD,
        help="Minimum average faithfulness required to pass (default: 0.8)",
    )
    parser.add_argument("--top-k", type=_positive_integer, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--candidate-k",
        type=_positive_integer,
        default=DEFAULT_RERANK_CANDIDATE_K,
    )
    parser.add_argument(
        "--min-relevance-score",
        type=_unit_interval,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
    )
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--evaluator-model", default=DEFAULT_GROQ_MODEL)
    return parser


def _config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        dataset=args.dataset,
        threshold=args.threshold,
        top_k=args.top_k,
        report_path=args.report,
        candidate_k=args.candidate_k,
        min_relevance_score=args.min_relevance_score,
        chroma_path=args.chroma_path,
        collection=args.collection,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        groq_model=args.groq_model,
        evaluator_model=args.evaluator_model,
    )


def build_evaluation_pipeline(config: EvaluationConfig) -> RAGPipeline:
    """Build the same Phase 2 retrieval and generation chain used by ask.py."""

    embedding_provider = SentenceTransformerEmbeddingProvider(
        config.embedding_model
    )
    vector_store = ChromaVectorStore(
        path=config.chroma_path,
        collection_name=config.collection,
    )
    retriever = build_phase2_retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker_model=config.reranker_model,
        candidate_k=config.candidate_k,
        min_relevance_score=config.min_relevance_score,
    )
    return RAGPipeline(
        retriever=retriever,
        language_model=GroqLanguageModel(model=config.groq_model),
    )


def build_evaluation_scorers(
    config: EvaluationConfig,
) -> tuple[MetricScorer, MetricScorer]:
    """Create Groq-judged Ragas metrics with local answer-relevancy embeddings."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to .env before evaluation."
        )
    try:
        from openai import OpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
    except ImportError as error:
        raise RuntimeError(
            "Phase 3 evaluation dependencies are missing. Install requirements.txt."
        ) from error

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    evaluator_llm = llm_factory(
        config.evaluator_model,
        provider="openai",
        client=client,
        temperature=0.0,
    )
    evaluator_embeddings = HuggingFaceEmbeddings(
        model=config.embedding_model,
        normalize_embeddings=True,
    )
    return create_ragas_scorers(
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
    )


def run_evaluation(
    config: EvaluationConfig,
    *,
    pipeline_factory: PipelineFactory | None = None,
    scorer_factory: ScorerFactory | None = None,
) -> ScoringReport:
    """Load, execute, and score every golden case."""

    cases = load_golden_dataset(config.dataset, min_cases=config.min_cases)
    resolved_pipeline_factory = pipeline_factory or build_evaluation_pipeline
    resolved_scorer_factory = scorer_factory or build_evaluation_scorers
    pipeline = resolved_pipeline_factory(config)
    faithfulness_scorer, answer_relevancy_scorer = resolved_scorer_factory(config)
    evaluation_results = evaluate_cases(
        cases,
        pipeline=pipeline,
        top_k=config.top_k,
    )
    return score_evaluation_results(
        evaluation_results,
        faithfulness_scorer=faithfulness_scorer,
        answer_relevancy_scorer=answer_relevancy_scorer,
    )


def evaluate_quality_gate(
    report: ScoringReport,
    *,
    threshold: float,
) -> QualityGateResult:
    """Require faithful answers, correct declines, and expected retrieval."""

    if isinstance(threshold, bool) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    average_faithfulness = report.average_faithfulness
    return QualityGateResult(
        report=report,
        threshold=float(threshold),
        faithfulness_passed=(
            average_faithfulness is not None
            and average_faithfulness >= threshold
        ),
        declines_passed=report.decline_accuracy == 1.0,
        retrieval_passed=report.retrieval_expectation_rate == 1.0,
    )


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def format_summary(gate: QualityGateResult) -> str:
    report = gate.report
    return "\n".join(
        (
            f"Evaluated cases: {len(report.results)}",
            f"Average faithfulness: {_percentage(report.average_faithfulness)} "
            f"(required: {gate.threshold:.2%})",
            "Average answer relevancy: "
            f"{_percentage(report.average_answer_relevancy)}",
            f"Decline accuracy: {_percentage(report.decline_accuracy)}",
            "Retrieval expectation rate: "
            f"{_percentage(report.retrieval_expectation_rate)}",
            f"Quality gate: {'PASS' if gate.passed else 'FAIL'}",
        )
    )


def main(
    argv: list[str] | None = None,
    *,
    evaluation_function: Callable[[EvaluationConfig], ScoringReport] | None = None,
) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError("python-dotenv is required for Phase 3 evaluation") from error

    load_dotenv(PROJECT_ROOT / ".env")
    config = _config_from_args(build_parser().parse_args(argv))
    report = (evaluation_function or run_evaluation)(config)
    gate = evaluate_quality_gate(report, threshold=config.threshold)
    write_markdown_report(config.report_path, render_markdown_report(gate))
    print(format_summary(gate))
    print(f"Report: {config.report_path}")
    return 0 if gate.passed else 1


def cli() -> int:
    try:
        return main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
