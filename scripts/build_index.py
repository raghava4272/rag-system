#!/usr/bin/env python3
"""Build or update the local Chroma index from source documents."""

from __future__ import annotations

import argparse
import math
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
from src.index_manifest import default_manifest_path
from src.pipeline import build_index


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


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
    parser.add_argument(
        "--lock-timeout-seconds",
        type=nonnegative_float,
        default=0.0,
        help=(
            "Seconds to wait for another index writer to finish; "
            "0 fails immediately (default: 0)"
        ),
    )
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
        manifest_path=default_manifest_path(args.chroma_path, args.collection),
        embedding_model=args.embedding_model,
        lock_timeout_seconds=args.lock_timeout_seconds,
    )
    print(
        f"Embedded {result.indexed_chunk_count} chunks from "
        f"{result.changed_source_file_count} changed source files; "
        f"reused {result.unchanged_source_file_count} unchanged source files; "
        f"removed {result.removed_source_file_count} deleted source files. "
        f"Index contains {result.total_chunk_count} total chunks from "
        f"{result.raw_document_count} document records across "
        f"{result.source_file_count} current source files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
