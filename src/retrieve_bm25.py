"""Standalone BM25 keyword retrieval over an in-memory chunk corpus."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.embed import Metadata


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class BM25Document:
    """A searchable text chunk and the metadata needed for citations."""

    id: str
    text: str
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class BM25Result:
    """One keyword-search result ordered by decreasing BM25 score."""

    id: str
    text: str
    metadata: Metadata
    score: float
    rank: int


def _tokenize(text: str) -> list[str]:
    """Normalize text into case-insensitive word tokens for BM25."""

    return TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """Rank chunks by exact-term relevance using Okapi BM25."""

    def __init__(self, documents: Sequence[BM25Document]) -> None:
        self.documents = list(documents)
        self._index = self._build_index(self.documents)

    @staticmethod
    def _build_index(documents: Sequence[BM25Document]) -> Any | None:
        if not documents:
            return None
        tokenized_corpus = [_tokenize(document.text) for document in documents]
        if any(not tokens for tokens in tokenized_corpus):
            raise ValueError("BM25 documents must contain searchable text")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise RuntimeError(
                "BM25 retrieval requires rank-bm25. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from error
        return BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[BM25Result]:
        """Return positive-score keyword matches in descending relevance order."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if allowed_sources is not None and not allowed_sources:
            return []
        if self._index is None:
            return []

        query_tokens = _tokenize(normalized_query)
        if not query_tokens:
            raise ValueError("query must contain searchable text")

        scores = self._index.get_scores(query_tokens)
        scored_documents = [
            (float(score), index, document)
            for index, (score, document) in enumerate(zip(scores, self.documents))
            if float(score) > 0.0
            and (
                allowed_sources is None
                or document.metadata.get("source") in allowed_sources
            )
        ]
        scored_documents.sort(key=lambda item: (-item[0], item[1]))

        return [
            BM25Result(
                id=document.id,
                text=document.text,
                metadata=dict(document.metadata),
                score=score,
                rank=rank,
            )
            for rank, (score, _, document) in enumerate(
                scored_documents[:top_k],
                start=1,
            )
        ]
