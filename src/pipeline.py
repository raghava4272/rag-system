"""Orchestration for index construction and retrieval-augmented answering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from src.chunk import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, Tokenizer
from src.embed import EmbeddingProvider, RetrievedChunk, VectorStore, index_documents
from src.generate import GeneratedAnswer, LanguageModel, generate_answer
from src.ingest import load_documents
from src.retrieve import DEFAULT_TOP_K


Token = TypeVar("Token")


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedChunk]: ...


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """Counts describing one completed local index build."""

    source_file_count: int
    raw_document_count: int
    indexed_chunk_count: int


@dataclass(frozen=True, slots=True)
class RAGRun:
    """One answer together with the evidence retrieved to produce it."""

    answer: GeneratedAnswer
    retrieved_chunks: tuple[RetrievedChunk, ...]


def build_index(
    source_directory: str | Path,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    tokenizer: Tokenizer[Token] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> IndexBuildResult:
    """Load a directory and persist embeddings for all supported source files."""

    documents = load_documents(source_directory)
    records = index_documents(
        documents,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return IndexBuildResult(
        source_file_count=len({document.source for document in documents}),
        raw_document_count=len(documents),
        indexed_chunk_count=len(records),
    )


class RAGPipeline:
    """Retrieve supporting chunks and generate one grounded answer."""

    def __init__(self, *, retriever: Retriever, language_model: LanguageModel) -> None:
        self.retriever = retriever
        self.language_model = language_model

    def ask(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> GeneratedAnswer:
        return self.run(question, top_k=top_k).answer

    def run(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> RAGRun:
        """Retrieve and answer while preserving contexts for evaluation."""

        chunks = tuple(self.retriever.retrieve(question, top_k=top_k))
        answer = generate_answer(
            question,
            list(chunks),
            language_model=self.language_model,
        )
        return RAGRun(answer=answer, retrieved_chunks=chunks)
