#!/usr/bin/env python3
"""Run the real-corpus Phase 2 retrieval regression gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ask import build_phase2_retriever
from src.embed import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
    SentenceTransformerEmbeddingProvider,
)
from src.regression import RegressionReport, RetrievalCase, evaluate_phase2
from src.rerank import (
    DEFAULT_MIN_RELEVANCE_SCORE,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
)
from src.retrieve import SemanticRetriever


DEFAULT_CASES_PATH = Path("data/phase2_regression.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 vector retrieval with the Phase 2 pipeline."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_RERANK_CANDIDATE_K)
    parser.add_argument(
        "--min-relevance-score",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
    )
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    return parser


def load_cases(path: Path) -> tuple[list[RetrievalCase], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        supported_payload = payload["supported"]
        unsupported_payload = payload["unsupported"]
        supported = [
            RetrievalCase(
                question=item["question"],
                expected_source_fragment=item["expected_source_fragment"],
            )
            for item in supported_payload
        ]
        unsupported = [str(question) for question in unsupported_payload]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load regression cases from {path}") from error
    return supported, unsupported


def format_report(report: RegressionReport) -> str:
    lines = [
        "Supported query source ranks:",
        "baseline -> phase2 | expected source | question",
    ]
    for result in report.supported_results:
        baseline = str(result.baseline_rank) if result.baseline_rank else "miss"
        upgraded = str(result.upgraded_rank) if result.upgraded_rank else "miss"
        lines.append(
            f"{baseline:>8} -> {upgraded:<6} | "
            f"{result.case.expected_source_fragment} | {result.case.question}"
        )

    lines.extend(("", "Unsupported query checks:"))
    for result in report.unsupported_results:
        status = "DECLINED" if result.declined else "FAILED"
        lines.append(f"{status}: {result.question}")

    lines.extend(
        (
            "",
            f"Vector Hit@1: {report.baseline_hit_at_1:.0%}",
            f"Phase 2 Hit@1: {report.upgraded_hit_at_1:.0%}",
            f"Vector Hit@3: {report.baseline_hit_at_3:.0%}",
            f"Phase 2 Hit@3: {report.upgraded_hit_at_3:.0%}",
            f"Unsupported decline rate: {report.unsupported_decline_rate:.0%}",
            f"No retrieval regression: {report.no_retrieval_regression}",
            f"Measurably improved: {report.measurably_improved}",
            f"Phase 2 ready: {report.phase2_ready}",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    supported, unsupported = load_cases(args.cases)
    embedding_provider = SentenceTransformerEmbeddingProvider(args.embedding_model)
    vector_store = ChromaVectorStore(
        path=args.chroma_path,
        collection_name=args.collection,
    )
    baseline = SemanticRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    upgraded = build_phase2_retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker_model=args.reranker_model,
        candidate_k=args.candidate_k,
        min_relevance_score=args.min_relevance_score,
    )
    report = evaluate_phase2(
        baseline_retriever=baseline,
        upgraded_retriever=upgraded,
        supported_cases=supported,
        unsupported_questions=unsupported,
        top_k=args.top_k,
    )
    print(format_report(report))
    return 0 if report.phase2_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
