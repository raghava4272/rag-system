"""Hybrid retrieval by reciprocal-rank fusion of vector and BM25 results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from src.embed import Metadata


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 10


class RankedChunk(Protocol):
    """Common fields required from any ranked retrieval result."""

    id: str
    text: str
    metadata: Metadata
    rank: int


class RankedRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> Sequence[RankedChunk]: ...


@dataclass(frozen=True, slots=True)
class HybridResult:
    """A chunk ranked by fused evidence from one or more retrievers."""

    id: str
    text: str
    metadata: Metadata
    rrf_score: float
    rank: int
    retrieved_by: tuple[str, ...]


@dataclass(slots=True)
class _FusionAccumulator:
    text: str
    metadata: Metadata
    score: float
    methods: list[str]


def reciprocal_rank_fusion(
    vector_results: Sequence[RankedChunk],
    bm25_results: Sequence[RankedChunk],
    *,
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridResult]:
    """Fuse rankings without comparing their incompatible raw score scales."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    fused: dict[str, _FusionAccumulator] = {}
    for method, results in (("vector", vector_results), ("bm25", bm25_results)):
        seen_ids: set[str] = set()
        for result in results:
            if not result.id:
                raise ValueError(f"{method} result id cannot be empty")
            if result.rank <= 0:
                raise ValueError(f"{method} result rank must be greater than zero")
            if result.id in seen_ids:
                raise ValueError(
                    f"{method} ranking contains duplicate chunk id {result.id!r}"
                )
            seen_ids.add(result.id)

            contribution = 1.0 / (rrf_k + result.rank)
            existing = fused.get(result.id)
            if existing is None:
                fused[result.id] = _FusionAccumulator(
                    text=result.text,
                    metadata=dict(result.metadata),
                    score=contribution,
                    methods=[method],
                )
                continue

            if existing.text != result.text or existing.metadata != result.metadata:
                raise ValueError(
                    f"Chunk id {result.id!r} has conflicting retrieval data"
                )
            existing.score += contribution
            existing.methods.append(method)

    ordered = sorted(
        fused.items(),
        key=lambda item: (-item[1].score, item[0]),
    )
    return [
        HybridResult(
            id=record_id,
            text=accumulator.text,
            metadata=dict(accumulator.metadata),
            rrf_score=accumulator.score,
            rank=rank,
            retrieved_by=tuple(accumulator.methods),
        )
        for rank, (record_id, accumulator) in enumerate(
            ordered[:top_k],
            start=1,
        )
    ]


class HybridRetriever:
    """Run semantic and keyword retrieval, then fuse their candidates."""

    def __init__(
        self,
        *,
        vector_retriever: RankedRetriever,
        bm25_retriever: RankedRetriever,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[HybridResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        vector_results = self.vector_retriever.retrieve(
            normalized_query,
            top_k=self.candidate_k,
            allowed_sources=allowed_sources,
        )
        bm25_results = self.bm25_retriever.retrieve(
            normalized_query,
            top_k=self.candidate_k,
            allowed_sources=allowed_sources,
        )
        return reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            top_k=top_k,
            rrf_k=self.rrf_k,
        )
