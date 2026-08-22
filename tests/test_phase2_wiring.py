"""Tests for loading the lexical corpus and wiring Phase 2 into the CLI."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from scripts.ask import build_phase2_retriever, main
from src.embed import ChromaVectorStore, StoredChunk
from src.generate import GeneratedAnswer
from src.hybrid_retrieve import HybridRetriever
from src.rerank import RerankingRetriever
from src.retrieve import SemanticRetriever
from src.retrieve_bm25 import BM25Retriever


class FakeCorpusCollection:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.get_options: dict[str, object] | None = None

    def get(self, **options: object) -> dict[str, object]:
        self.get_options = options
        return self.response

    def query(self, **options: object) -> dict[str, object]:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def upsert(self, **payload: object) -> None:
        pass


class FakeChromaClient:
    def __init__(self, collection: FakeCorpusCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, **options: object) -> FakeCorpusCollection:
        return self.collection


class FakeVectorStore:
    def __init__(self, chunks: list[StoredChunk]) -> None:
        self.chunks = chunks

    def all_chunks(self) -> list[StoredChunk]:
        return self.chunks

    def search(self, query_embedding: list[float], *, top_k: int) -> list[object]:
        return []


class Phase2CorpusLoadingTests(unittest.TestCase):
    def test_chroma_returns_all_stored_chunks_for_keyword_indexing(self) -> None:
        collection = FakeCorpusCollection(
            {
                "ids": ["chunk-a", "chunk-b"],
                "documents": ["First chunk", "Second chunk"],
                "metadatas": [
                    {"source": "guide.md", "chunk_index": 0},
                    {"source": "paper.pdf", "page_number": 2},
                ],
            }
        )
        store = ChromaVectorStore(client=FakeChromaClient(collection))

        chunks = store.all_chunks()

        self.assertEqual(
            collection.get_options,
            {"include": ["documents", "metadatas"]},
        )
        self.assertEqual(
            chunks,
            [
                StoredChunk(
                    id="chunk-a",
                    text="First chunk",
                    metadata={"source": "guide.md", "chunk_index": 0},
                ),
                StoredChunk(
                    id="chunk-b",
                    text="Second chunk",
                    metadata={"source": "paper.pdf", "page_number": 2},
                ),
            ],
        )

    def test_malformed_chroma_corpus_columns_are_rejected(self) -> None:
        collection = FakeCorpusCollection(
            {
                "ids": ["chunk-a", "chunk-b"],
                "documents": ["Only one document"],
                "metadatas": [{}, {}],
            }
        )
        store = ChromaVectorStore(client=FakeChromaClient(collection))

        with self.assertRaisesRegex(ValueError, "malformed"):
            store.all_chunks()


class Phase2RetrieverFactoryTests(unittest.TestCase):
    def test_factory_connects_semantic_bm25_hybrid_rerank_and_guard(self) -> None:
        chunks = [
            StoredChunk(
                id=f"chunk-{index}",
                text=text,
                metadata={"source": f"source-{index}.md", "chunk_index": 0},
            )
            for index, text in enumerate(
                (
                    "Chroma stores vectors",
                    "BM25 matches keywords",
                    "Cross encoders rerank results",
                )
            )
        ]
        embedding_provider = object()
        vector_store = FakeVectorStore(chunks)
        reranker = object()

        retriever = build_phase2_retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            reranker=reranker,
            candidate_k=12,
            min_relevance_score=0.65,
        )

        self.assertIsInstance(retriever, RerankingRetriever)
        self.assertEqual(retriever.candidate_k, 12)
        self.assertEqual(retriever.min_relevance_score, 0.65)
        self.assertIs(retriever.reranker, reranker)
        self.assertIsInstance(retriever.hybrid_retriever, HybridRetriever)
        self.assertIsInstance(
            retriever.hybrid_retriever.vector_retriever,
            SemanticRetriever,
        )
        self.assertIs(
            retriever.hybrid_retriever.vector_retriever.embedding_provider,
            embedding_provider,
        )
        self.assertIsInstance(
            retriever.hybrid_retriever.bm25_retriever,
            BM25Retriever,
        )
        self.assertEqual(
            [document.id for document in retriever.hybrid_retriever.bm25_retriever.documents],
            ["chunk-0", "chunk-1", "chunk-2"],
        )

    def test_cli_uses_phase2_factory_with_configurable_safety_options(self) -> None:
        with (
            patch("scripts.ask.GroqLanguageModel") as language_model_class,
            patch(
                "scripts.ask.SentenceTransformerEmbeddingProvider"
            ) as embedding_provider_class,
            patch("scripts.ask.ChromaVectorStore") as vector_store_class,
            patch("scripts.ask.build_phase2_retriever") as retriever_factory,
            patch("scripts.ask.RAGPipeline") as pipeline_class,
        ):
            retriever = object()
            retriever_factory.return_value = retriever
            pipeline_class.return_value.ask.return_value = GeneratedAnswer(
                text="Safe answer [1]",
                citations=[],
            )

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main([
                    "How does retrieval work?",
                    "--candidate-k",
                    "12",
                    "--min-relevance-score",
                    "0.65",
                    "--reranker-model",
                    "local/test-reranker",
                ])

        self.assertEqual(exit_code, 0)
        retriever_factory.assert_called_once_with(
            embedding_provider=embedding_provider_class.return_value,
            vector_store=vector_store_class.return_value,
            reranker_model="local/test-reranker",
            candidate_k=12,
            min_relevance_score=0.65,
        )
        pipeline_class.assert_called_once_with(
            retriever=retriever,
            language_model=language_model_class.return_value,
        )
        self.assertIn("Safe answer", output.getvalue())


if __name__ == "__main__":
    unittest.main()
