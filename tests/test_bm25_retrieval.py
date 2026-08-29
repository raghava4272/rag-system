"""Behavioral tests for standalone BM25 keyword retrieval."""

from __future__ import annotations

import unittest

from src.retrieve_bm25 import BM25Document, BM25Retriever


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            BM25Document(
                id="chunk-chroma",
                text="ChromaDB persistence stores embedding vectors on disk.",
                metadata={"source": "vectorstores.md", "chunk_index": 2},
            ),
            BM25Document(
                id="chunk-splitting",
                text="Recursive text splitting keeps useful context overlap.",
                metadata={"source": "splitters.md", "chunk_index": 1},
            ),
            BM25Document(
                id="chunk-reranking",
                text="A cross encoder reranks retrieval candidates for relevance.",
                metadata={"source": "reranking.md", "chunk_index": 0},
            ),
        ]
        self.retriever = BM25Retriever(self.documents)

    def test_exact_keyword_match_ranks_first_and_preserves_citation_data(self) -> None:
        results = self.retriever.retrieve("How does CHROMADB persistence work?", top_k=2)

        self.assertEqual(results[0].id, "chunk-chroma")
        self.assertEqual(results[0].text, self.documents[0].text)
        self.assertEqual(results[0].metadata, self.documents[0].metadata)
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].score, 0.0)

    def test_token_matching_is_case_insensitive_and_ignores_punctuation(self) -> None:
        plain = self.retriever.retrieve("chromadb", top_k=1)
        punctuated = self.retriever.retrieve("  CHROMADB!!!  ", top_k=1)

        self.assertEqual(plain, punctuated)
        self.assertEqual(plain[0].id, "chunk-chroma")

    def test_top_k_limits_results_and_assigns_consecutive_ranks(self) -> None:
        results = self.retriever.retrieve("text splitting context", top_k=2)

        self.assertLessEqual(len(results), 2)
        self.assertEqual(
            [result.rank for result in results],
            list(range(1, len(results) + 1)),
        )

    def test_query_without_keyword_matches_returns_no_results(self) -> None:
        self.assertEqual(
            self.retriever.retrieve("photosynthesis chlorophyll", top_k=3),
            [],
        )

    def test_empty_corpus_returns_no_results(self) -> None:
        retriever = BM25Retriever([])

        self.assertEqual(retriever.retrieve("retrieval", top_k=5), [])

    def test_permission_scope_filters_documents_before_keyword_ranking(self) -> None:
        results = self.retriever.retrieve(
            "retrieval candidates relevance",
            top_k=3,
            allowed_sources=frozenset({"reranking.md"}),
        )

        self.assertEqual([result.id for result in results], ["chunk-reranking"])
        self.assertEqual(results[0].rank, 1)

    def test_empty_permission_scope_returns_no_keyword_results(self) -> None:
        self.assertEqual(
            self.retriever.retrieve(
                "chromadb",
                top_k=3,
                allowed_sources=frozenset(),
            ),
            [],
        )

    def test_blank_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query"):
            self.retriever.retrieve("  \n", top_k=3)

    def test_nonpositive_top_k_is_rejected(self) -> None:
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    self.retriever.retrieve("retrieval", top_k=top_k)


if __name__ == "__main__":
    unittest.main()
