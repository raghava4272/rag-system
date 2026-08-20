#!/usr/bin/env python3
"""Build or update the local Chroma index from source documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunk import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from src.embed import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
    SentenceTransformerEmbeddingProvider,
)
from src.pipeline import build_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest source documents and build the persistent vector index."
    )
    parser.add_argument(
        "source_directory",
        nargs="?",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing Markdown, MDX, and PDF sources (default: data/raw)",
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
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embedding_provider = SentenceTransformerEmbeddingProvider(args.embedding_model)
    vector_store = ChromaVectorStore(
        path=args.chroma_path,
        collection_name=args.collection,
    )
    result = build_index(
        args.source_directory,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(
        f"Indexed {result.indexed_chunk_count} chunks from "
        f"{result.raw_document_count} document records across "
        f"{result.source_file_count} source files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

