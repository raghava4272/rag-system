"""Top-k semantic retrieval over an embedded vector store."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.embed import EmbeddingProvider, RetrievedChunk


DEFAULT_TOP_K = 5


class SearchableVectorStore(Protocol):
    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[RetrievedChunk]: ...


class SemanticRetriever:
    """Embed a natural-language query and retrieve its nearest chunks."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: SearchableVectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if allowed_sources is not None and not allowed_sources:
            return []

        query_embedding = self.embedding_provider.embed_query(normalized_query)
        if not query_embedding:
            raise ValueError("The embedding provider returned an empty query embedding")
        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            allowed_sources=allowed_sources,
        )
