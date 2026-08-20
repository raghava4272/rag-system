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
from src.llm import DEFAULT_GROQ_MODEL, GroqLanguageModel
from src.pipeline import RAGPipeline
from src.retrieve import DEFAULT_TOP_K, SemanticRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a question and receive an answer grounded in indexed sources."
    )
    parser.add_argument("question", help="Question to answer from the indexed corpus")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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
    retriever = SemanticRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
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
