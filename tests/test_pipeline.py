"""Integration-style tests for the Phase 1 RAG pipeline."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from src.embed import RetrievedChunk
from src.generate import NO_CONTEXT_ANSWER
from src.pipeline import RAGPipeline, build_index


class WhitespaceTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.document_calls.append(batch)
        return [[float(len(text.split())), 1.0] for text in batch]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: list[object] = []

    def upsert(self, records: Sequence[object]) -> None:
        self.records.extend(records)


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, frozenset[str] | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_sources: frozenset[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, allowed_sources))
        return self.results


class FakeLanguageModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class BuildIndexPipelineTests(unittest.TestCase):
    def test_build_index_loads_chunks_embeds_and_stores_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_directory = Path(temporary_directory)
            (source_directory / "guide.md").write_text(
                "retrieval evidence " * 450,
                encoding="utf-8",
            )
            embedder = FakeEmbeddingProvider()
            store = FakeVectorStore()

            result = build_index(
                source_directory,
                embedding_provider=embedder,
                vector_store=store,
                tokenizer=WhitespaceTokenizer(),
                chunk_size=500,
                overlap=100,
            )

        self.assertEqual(result.source_file_count, 1)
        self.assertEqual(result.raw_document_count, 1)
        self.assertEqual(result.indexed_chunk_count, 2)
        self.assertEqual(len(embedder.document_calls[0]), 2)
        self.assertEqual(len(store.records), 2)


class QuestionAnswerPipelineTests(unittest.TestCase):
    def test_ask_retrieves_then_generates_a_cited_answer(self) -> None:
        chunks = [
            RetrievedChunk(
                id="chunk-1",
                text="RAG retrieves relevant evidence.",
                metadata={"source": "guide.md", "chunk_index": 0},
                distance=0.1,
                rank=1,
            )
        ]
        retriever = FakeRetriever(chunks)
        model = FakeLanguageModel("RAG retrieves evidence before answering [1].")
        pipeline = RAGPipeline(retriever=retriever, language_model=model)

        answer = pipeline.ask("How does RAG work?", top_k=4)

        self.assertEqual(retriever.calls, [("How does RAG work?", 4, None)])
        self.assertEqual(answer.text, model.response)
        self.assertEqual(answer.citations[0].source, "guide.md")
        self.assertEqual(len(model.prompts), 1)

    def test_ask_with_no_results_returns_safe_answer_without_model_call(self) -> None:
        retriever = FakeRetriever([])
        model = FakeLanguageModel("should not be used")
        pipeline = RAGPipeline(retriever=retriever, language_model=model)

        answer = pipeline.ask("Unknown question")

        self.assertEqual(answer.text, NO_CONTEXT_ANSWER)
        self.assertEqual(answer.citations, [])
        self.assertEqual(model.prompts, [])

    def test_permission_scope_is_forwarded_and_checked_before_generation(self) -> None:
        allowed = frozenset({"team-a.md"})
        chunks = [
            RetrievedChunk(
                id="team-a:0",
                text="Team A evidence.",
                metadata={"source": "team-a.md", "chunk_index": 0},
                distance=0.1,
                rank=1,
            )
        ]
        retriever = FakeRetriever(chunks)
        model = FakeLanguageModel("Team A evidence is available [1].")
        pipeline = RAGPipeline(retriever=retriever, language_model=model)

        pipeline.ask("What can Team A see?", allowed_sources=allowed)

        self.assertEqual(retriever.calls, [("What can Team A see?", 5, allowed)])
        self.assertEqual(len(model.prompts), 1)

    def test_pipeline_blocks_out_of_scope_chunk_before_it_reaches_the_model(self) -> None:
        retriever = FakeRetriever([
            RetrievedChunk(
                id="team-b:0",
                text="Team B secret.",
                metadata={"source": "team-b.md", "chunk_index": 0},
                distance=0.1,
                rank=1,
            )
        ])
        model = FakeLanguageModel("must not run")
        pipeline = RAGPipeline(retriever=retriever, language_model=model)

        with self.assertRaises(PermissionError):
            pipeline.ask(
                "What is Team B's secret?",
                allowed_sources=frozenset({"team-a.md"}),
            )

        self.assertEqual(model.prompts, [])


if __name__ == "__main__":
    unittest.main()
