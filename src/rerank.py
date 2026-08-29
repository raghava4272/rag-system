"""Cross-encoder reranking for hybrid retrieval candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.embed import Metadata
from src.hybrid_retrieve import HybridResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANK_CANDIDATE_K = 10
DEFAULT_MIN_RELEVANCE_SCORE = 0.5


@dataclass(frozen=True, slots=True)
class RerankedResult:
    """A hybrid candidate reordered by direct query-to-chunk relevance."""

    id: str
    text: str
    metadata: Metadata
    relevance_score: float
    rank: int
    rrf_score: float
    retrieved_by: tuple[str, ...]


class HybridCandidateRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> Sequence[HybridResult]: ...


class CandidateReranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridResult],
        *,
        top_k: int,
    ) -> Sequence[RerankedResult]: ...


class CrossEncoderReranker:
    """Score query/chunk pairs with a local Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                from torch.nn import Sigmoid
            except ImportError as error:
                raise RuntimeError(
                    "Cross-encoder reranking requires sentence-transformers and "
                    "torch. Install dependencies with "
                    "`python3 -m pip install -r requirements.txt`."
                ) from error

            self._model = CrossEncoder(
                self.model_name,
                activation_fn=Sigmoid(),
            )
        return self._model

    @staticmethod
    def _as_scores(value: Any) -> list[float]:
        raw_scores = value.tolist() if hasattr(value, "tolist") else value
        try:
            scores = [float(score) for score in raw_scores]
        except (TypeError, ValueError) as error:
            raise ValueError("The cross-encoder returned malformed scores") from error
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("The cross-encoder returned a non-finite score")
        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridResult],
        *,
        top_k: int,
    ) -> list[RerankedResult]:
        """Score all candidates jointly with the query and return the best ones."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not candidates:
            return []

        pairs = [(normalized_query, candidate.text) for candidate in candidates]
        scores = self._as_scores(
            self._get_model().predict(pairs, show_progress_bar=False)
        )
        if len(scores) != len(candidates):
            raise ValueError(
                "The cross-encoder score count does not match the candidate count"
            )

        ordered = sorted(
            zip(scores, range(len(candidates)), candidates),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            RerankedResult(
                id=candidate.id,
                text=candidate.text,
                metadata=dict(candidate.metadata),
                relevance_score=score,
                rank=rank,
                rrf_score=candidate.rrf_score,
                retrieved_by=candidate.retrieved_by,
            )
            for rank, (score, _, candidate) in enumerate(
                ordered[:top_k],
                start=1,
            )
        ]


class RerankingRetriever:
    """Run hybrid retrieval, rerank candidates, and reject weak evidence."""

    def __init__(
        self,
        *,
        hybrid_retriever: HybridCandidateRetriever,
        reranker: CandidateReranker,
        candidate_k: int = DEFAULT_RERANK_CANDIDATE_K,
        min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between zero and one")
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.min_relevance_score = min_relevance_score

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[RerankedResult]:
        """Return only reranked chunks with enough direct query relevance."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        candidates = list(
            self.hybrid_retriever.retrieve(
                normalized_query,
                top_k=self.candidate_k,
                allowed_sources=allowed_sources,
            )
        )
        if not candidates:
            return []

        reranked = self.reranker.rerank(
            normalized_query,
            candidates,
            top_k=top_k,
        )
        return [
            result
            for result in reranked
            if result.relevance_score >= self.min_relevance_score
        ]
