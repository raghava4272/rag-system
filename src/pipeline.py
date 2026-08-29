"""Orchestration for index construction and retrieval-augmented answering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from src.chunk import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, Tokenizer
from src.embed import EmbeddingProvider, RetrievedChunk, VectorStore, index_documents
from src.generate import GeneratedAnswer, LanguageModel, generate_answer
from src.index_manifest import (
    IndexConfiguration,
    IndexManifest,
    SourceManifest,
    acquire_index_lock,
    fingerprint_file,
    load_manifest,
    write_manifest,
)
from src.ingest import discover_source_files, load_document, load_documents
from src.retrieve import DEFAULT_TOP_K


Token = TypeVar("Token")


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> Sequence[RetrievedChunk]: ...


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """Counts describing one completed local index build."""

    source_file_count: int
    raw_document_count: int
    indexed_chunk_count: int
    changed_source_file_count: int = 0
    unchanged_source_file_count: int = 0
    removed_source_file_count: int = 0
    total_chunk_count: int = 0


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
    manifest_path: str | Path | None = None,
    embedding_model: str | None = None,
    lock_timeout_seconds: float = 0,
) -> IndexBuildResult:
    """Load and persist source embeddings, optionally using an incremental manifest."""

    if manifest_path is not None:
        resolved_manifest_path = Path(manifest_path)
        with acquire_index_lock(
            resolved_manifest_path,
            timeout_seconds=lock_timeout_seconds,
        ):
            return _build_incremental_index(
                source_directory,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
                manifest_path=resolved_manifest_path,
                embedding_model=(
                    embedding_model
                    or str(getattr(embedding_provider, "model_name", "unknown"))
                ),
            )

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
        changed_source_file_count=len({document.source for document in documents}),
        total_chunk_count=len(records),
    )


def _tokenizer_identity(tokenizer: object | None) -> str:
    if tokenizer is None:
        return "tiktoken:cl100k_base"
    tokenizer_type = type(tokenizer)
    return f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}"


def _build_incremental_index(
    source_directory: str | Path,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    tokenizer: Tokenizer[Token] | None,
    chunk_size: int,
    overlap: int,
    manifest_path: Path,
    embedding_model: str,
) -> IndexBuildResult:
    paths = discover_source_files(source_directory)
    current_fingerprints = {str(path): fingerprint_file(path) for path in paths}
    configuration = IndexConfiguration(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
        tokenizer=_tokenizer_identity(tokenizer),
    )
    previous = load_manifest(manifest_path)
    stored_ids = {chunk.id for chunk in vector_store.all_chunks()}
    previous_sources = previous.sources if previous is not None else {}
    current_sources = set(current_fingerprints)
    removed_sources = set(previous_sources) - current_sources

    if previous is None or previous.configuration != configuration:
        changed_sources = current_sources
        unchanged_sources: set[str] = set()
        obsolete_ids = set(stored_ids)
    else:
        expected_ids = {
            record_id
            for state in previous_sources.values()
            for record_id in state.chunk_ids
        }
        changed_sources = {
            source
            for source in current_sources
            if source not in previous_sources
            or previous_sources[source].fingerprint != current_fingerprints[source]
            or not set(previous_sources[source].chunk_ids).issubset(stored_ids)
        }
        unchanged_sources = current_sources - changed_sources
        obsolete_ids = stored_ids - expected_ids
        for source in changed_sources | removed_sources:
            state = previous_sources.get(source)
            if state is not None:
                obsolete_ids.update(state.chunk_ids)

    documents = []
    raw_counts: dict[str, int] = {}
    for path in paths:
        source = str(path)
        if source not in changed_sources:
            continue
        loaded = load_document(path)
        documents.extend(loaded)
        raw_counts[source] = len(loaded)

    records = index_documents(
        documents,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    new_ids = {record.id for record in records}
    ids_by_source: dict[str, list[str]] = {source: [] for source in changed_sources}
    for record in records:
        ids_by_source[str(record.metadata["source"])].append(record.id)

    delete_ids = sorted(obsolete_ids - new_ids)
    if delete_ids:
        vector_store.delete(delete_ids)

    updated_sources = {
        source: previous_sources[source]
        for source in unchanged_sources
    }
    for source in changed_sources:
        updated_sources[source] = SourceManifest(
            fingerprint=current_fingerprints[source],
            raw_document_count=raw_counts.get(source, 0),
            chunk_ids=tuple(ids_by_source[source]),
        )
    updated_manifest = IndexManifest(
        configuration=configuration,
        sources=updated_sources,
    )
    write_manifest(manifest_path, updated_manifest)

    return IndexBuildResult(
        source_file_count=len(current_sources),
        raw_document_count=sum(
            state.raw_document_count for state in updated_sources.values()
        ),
        indexed_chunk_count=len(records),
        changed_source_file_count=len(changed_sources),
        unchanged_source_file_count=len(unchanged_sources),
        removed_source_file_count=len(removed_sources),
        total_chunk_count=sum(len(state.chunk_ids) for state in updated_sources.values()),
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
        allowed_sources: frozenset[str] | None = None,
    ) -> GeneratedAnswer:
        return self.run(
            question,
            top_k=top_k,
            allowed_sources=allowed_sources,
        ).answer

    def run(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        allowed_sources: frozenset[str] | None = None,
    ) -> RAGRun:
        """Retrieve and answer while preserving contexts for evaluation."""

        if allowed_sources is None:
            chunks = tuple(self.retriever.retrieve(question, top_k=top_k))
        else:
            chunks = tuple(
                self.retriever.retrieve(
                    question,
                    top_k=top_k,
                    allowed_sources=allowed_sources,
                )
            )
            if any(
                chunk.metadata.get("source") not in allowed_sources
                for chunk in chunks
            ):
                raise PermissionError("Retriever returned evidence outside its scope")
        answer = generate_answer(
            question,
            list(chunks),
            language_model=self.language_model,
        )
        return RAGRun(answer=answer, retrieved_chunks=chunks)
