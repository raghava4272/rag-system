"""Behavioral tests for top-k semantic retrieval."""

from __future__ import annotations

import unittest
from collections.abc import Sequence

from src.embed import ChromaVectorStore, RetrievedChunk
from src.retrieve import SemanticRetriever


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.25, 0.75]


class FakeSearchStore:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.search_calls: list[tuple[list[float], int, frozenset[str] | None]] = []

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.search_calls.append((list(query_embedding), top_k, allowed_sources))
        return self.results


class FakeQueryCollection:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.query_options: dict[str, object] | None = None

    def query(self, **options: object) -> dict[str, object]:
        self.query_options = options
        return self.response

    def upsert(self, **payload: object) -> None:
        pass


class FakeChromaClient:
    def __init__(self, collection: FakeQueryCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, **options: object) -> FakeQueryCollection:
        return self.collection


class SemanticRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbeddingProvider()
        self.expected = [
            RetrievedChunk(
                id="chunk-1",
                text="Relevant evidence",
                metadata={"source": "guide.md", "chunk_index": 0},
                distance=0.12,
                rank=1,
            )
        ]
        self.store = FakeSearchStore(self.expected)
        self.retriever = SemanticRetriever(
            embedding_provider=self.embedder,
            vector_store=self.store,
        )

    def test_query_is_embedded_and_forwarded_with_top_k(self) -> None:
        results = self.retriever.retrieve("How does retrieval work?", top_k=3)

        self.assertEqual(results, self.expected)
        self.assertEqual(self.embedder.query_calls, ["How does retrieval work?"])
        self.assertEqual(self.store.search_calls, [([0.25, 0.75], 3, None)])

    def test_allowed_sources_are_applied_inside_vector_search(self) -> None:
        allowed = frozenset({"data/raw/team-a.md"})

        results = self.retriever.retrieve(
            "private question",
            top_k=2,
            allowed_sources=allowed,
        )

        self.assertEqual(results, self.expected)
        self.assertEqual(self.store.search_calls, [([0.25, 0.75], 2, allowed)])

    def test_empty_permission_scope_skips_embedding_and_storage(self) -> None:
        results = self.retriever.retrieve(
            "private question",
            top_k=2,
            allowed_sources=frozenset(),
        )

        self.assertEqual(results, [])
        self.assertEqual(self.embedder.query_calls, [])
        self.assertEqual(self.store.search_calls, [])

    def test_blank_query_is_rejected_without_external_calls(self) -> None:
        with self.assertRaisesRegex(ValueError, "query"):
            self.retriever.retrieve("  \n")

        self.assertEqual(self.embedder.query_calls, [])
        self.assertEqual(self.store.search_calls, [])

    def test_nonpositive_top_k_is_rejected_without_external_calls(self) -> None:
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    self.retriever.retrieve("question", top_k=top_k)

        self.assertEqual(self.embedder.query_calls, [])
        self.assertEqual(self.store.search_calls, [])


class ChromaSearchAdapterTests(unittest.TestCase):
    def test_chroma_results_are_converted_to_ranked_chunks(self) -> None:
        collection = FakeQueryCollection(
            {
                "ids": [["chunk-a", "chunk-b"]],
                "documents": [["First", "Second"]],
                "metadatas": [[
                    {"source": "paper.pdf", "page_number": 4},
                    {"source": "guide.md", "chunk_index": 2},
                ]],
                "distances": [[0.1, 0.35]],
            }
        )
        store = ChromaVectorStore(client=FakeChromaClient(collection))

        results = store.search([0.25, 0.75], top_k=2)

        self.assertEqual(
            collection.query_options,
            {
                "query_embeddings": [[0.25, 0.75]],
                "n_results": 2,
                "include": ["documents", "metadatas", "distances"],
            },
        )
        self.assertEqual(
            results,
            [
                RetrievedChunk(
                    id="chunk-a",
                    text="First",
                    metadata={"source": "paper.pdf", "page_number": 4},
                    distance=0.1,
                    rank=1,
                ),
                RetrievedChunk(
                    id="chunk-b",
                    text="Second",
                    metadata={"source": "guide.md", "chunk_index": 2},
                    distance=0.35,
                    rank=2,
                ),
            ],
        )

    def test_empty_chroma_result_returns_empty_list(self) -> None:
        collection = FakeQueryCollection(
            {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        )
        store = ChromaVectorStore(client=FakeChromaClient(collection))

        self.assertEqual(store.search([1.0, 0.0], top_k=5), [])

    def test_chroma_uses_an_exact_source_filter_for_authorized_search(self) -> None:
        collection = FakeQueryCollection(
            {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        )
        store = ChromaVectorStore(client=FakeChromaClient(collection))

        store.search(
            [1.0, 0.0],
            top_k=5,
            allowed_sources=frozenset({"data/raw/b.md", "data/raw/a.md"}),
        )

        self.assertEqual(
            collection.query_options["where"],
            {"source": {"$in": ["data/raw/a.md", "data/raw/b.md"]}},
        )

    def test_malformed_chroma_columns_are_rejected(self) -> None:
        collection = FakeQueryCollection(
            {
                "ids": [["chunk-a", "chunk-b"]],
                "documents": [["Only one document"]],
                "metadatas": [[{}, {}]],
                "distances": [[0.1, 0.2]],
            }
        )
        store = ChromaVectorStore(client=FakeChromaClient(collection))

        with self.assertRaisesRegex(ValueError, "malformed"):
            store.search([1.0, 0.0], top_k=2)


if __name__ == "__main__":
    unittest.main()
