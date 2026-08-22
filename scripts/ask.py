#!/usr/bin/env python3
"""Ask a grounded question over the persistent local vector index."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embed import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
    SentenceTransformerEmbeddingProvider,
)
from src.generate import GeneratedAnswer
from src.hybrid_retrieve import HybridRetriever
from src.llm import DEFAULT_GROQ_MODEL, GroqLanguageModel
from src.pipeline import RAGPipeline
from src.rerank import (
    DEFAULT_MIN_RELEVANCE_SCORE,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
    CandidateReranker,
    CrossEncoderReranker,
    RerankingRetriever,
)
from src.retrieve import DEFAULT_TOP_K, SemanticRetriever
from src.retrieve_bm25 import BM25Document, BM25Retriever


def build_phase2_retriever(
    *,
    embedding_provider: SentenceTransformerEmbeddingProvider,
    vector_store: ChromaVectorStore,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    candidate_k: int = DEFAULT_RERANK_CANDIDATE_K,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    reranker: CandidateReranker | None = None,
) -> RerankingRetriever:
    """Build the hybrid, reranked, confidence-filtered retrieval chain."""

    semantic_retriever = SemanticRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    bm25_documents = [
        BM25Document(
            id=chunk.id,
            text=chunk.text,
            metadata=dict(chunk.metadata),
        )
        for chunk in vector_store.all_chunks()
    ]
    bm25_retriever = BM25Retriever(bm25_documents)
    hybrid_retriever = HybridRetriever(
        vector_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
        candidate_k=candidate_k,
    )
    resolved_reranker = reranker or CrossEncoderReranker(reranker_model)
    return RerankingRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=resolved_reranker,
        candidate_k=candidate_k,
        min_relevance_score=min_relevance_score,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a question and receive an answer grounded in indexed sources."
    )
    parser.add_argument("question", help="Question to answer from the indexed corpus")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_RERANK_CANDIDATE_K,
        help=(
            "Hybrid candidates to send to the reranker "
            f"(default: {DEFAULT_RERANK_CANDIDATE_K})"
        ),
    )
    parser.add_argument(
        "--min-relevance-score",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
        help=(
            "Minimum cross-encoder score required for evidence "
            f"(default: {DEFAULT_MIN_RELEVANCE_SCORE})"
        ),
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=DEFAULT_CHROMA_PATH,
        help=f"Persistent Chroma directory (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence Transformers model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--groq-model",
        default=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        help=f"Groq generation model (default: {DEFAULT_GROQ_MODEL})",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Local cross-encoder model (default: {DEFAULT_RERANKER_MODEL})",
    )
    return parser


def format_answer(answer: GeneratedAnswer) -> str:
    if not answer.citations:
        return answer.text

    source_lines: list[str] = []
    for citation in answer.citations:
        location: list[str] = []
        if citation.page_number is not None:
            location.append(f"page {citation.page_number}")
        if citation.chunk_index is not None:
            location.append(f"chunk {citation.chunk_index}")
        suffix = f" ({', '.join(location)})" if location else ""
        source_lines.append(
            f"[{citation.number}] {citation.source}{suffix}"
        )
    return f"{answer.text}\n\nSources:\n" + "\n".join(source_lines)


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "The ask CLI requires python-dotenv. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from error

    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args(argv)

    language_model = GroqLanguageModel(model=args.groq_model)
    embedding_provider = SentenceTransformerEmbeddingProvider(args.embedding_model)
    vector_store = ChromaVectorStore(
        path=args.chroma_path,
        collection_name=args.collection,
    )
    retriever = build_phase2_retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker_model=args.reranker_model,
        candidate_k=args.candidate_k,
        min_relevance_score=args.min_relevance_score,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        language_model=language_model,
    )
    answer = pipeline.ask(args.question, top_k=args.top_k)
    print(format_answer(answer))
    return 0


def cli() -> int:
    try:
        return main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
