"""Behavioral tests for embedding chunks and persisting them in Chroma."""

from __future__ import annotations

import unittest
from collections.abc import Sequence

from src.embed import (
    ChromaVectorStore,
    SentenceTransformerEmbeddingProvider,
    index_documents,
)
from src.ingest import RawDocument


class WhitespaceTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class FakeEmbeddingProvider:
    def __init__(self, *, return_too_few: bool = False) -> None:
        self.document_calls: list[list[str]] = []
        self.return_too_few = return_too_few

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.document_calls.append(batch)
        vectors = [
            [float(len(text.split())), float(index)]
            for index, text in enumerate(batch)
        ]
        return vectors[:-1] if self.return_too_few else vectors

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text.split())), 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upsert_calls: list[list[object]] = []

    def upsert(self, records: Sequence[object]) -> None:
        self.upsert_calls.append(list(records))


class FakeCollection:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None
        self.deleted_ids: list[str] | None = None

    def upsert(self, **payload: object) -> None:
        self.payload = payload

    def delete(self, *, ids: list[str]) -> None:
        self.deleted_ids = ids


class FakeChromaClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.collection_options: dict[str, object] | None = None

    def get_or_create_collection(self, **options: object) -> FakeCollection:
        self.collection_options = options
        return self.collection


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.document_options: dict[str, object] | None = None
        self.query_options: dict[str, object] | None = None

    def encode_document(self, texts: Sequence[str], **options: object) -> list[list[float]]:
        self.document_options = options
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def encode_query(self, text: str, **options: object) -> list[float]:
        self.query_options = options
        return [float(len(text)), 1.0]


def make_document(token_count: int) -> str:
    return " ".join(f"token-{index}" for index in range(token_count))


class EmbeddingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()
        self.embedder = FakeEmbeddingProvider()
        self.store = FakeVectorStore()

    def test_documents_are_chunked_embedded_and_upserted_with_metadata(self) -> None:
        source = RawDocument(
            text=make_document(900),
            source="data/raw/guide.md",
            document_type="markdown",
        )

        records = index_documents(
            [source],
            embedding_provider=self.embedder,
            vector_store=self.store,
            tokenizer=self.tokenizer,
            chunk_size=500,
            overlap=100,
        )

        self.assertEqual(len(records), 2)
        embedded_token_counts = [
            len(text.split()) for text in self.embedder.document_calls[0]
        ]
        self.assertEqual(embedded_token_counts, [500, 500])
        self.assertEqual(records[0].embedding, [500.0, 0.0])
        self.assertEqual(records[1].metadata["chunk_index"], 1)
        self.assertEqual(records[1].metadata["token_start"], 400)
        self.assertEqual(records[1].metadata["token_end"], 900)
        self.assertEqual(records[0].metadata["source"], "data/raw/guide.md")
        self.assertNotIn("page_number", records[0].metadata)
        self.assertEqual(self.store.upsert_calls, [records])

    def test_pdf_page_number_is_preserved_for_citations(self) -> None:
        source = RawDocument(
            text="Page-specific evidence",
            source="data/raw/paper.pdf",
            document_type="pdf",
            page_number=7,
        )

        records = index_documents(
            [source],
            embedding_provider=self.embedder,
            vector_store=self.store,
            tokenizer=self.tokenizer,
        )

        self.assertEqual(records[0].metadata["page_number"], 7)

    def test_record_ids_are_stable_and_content_sensitive(self) -> None:
        original = RawDocument("same text", "guide.md", "markdown")
        changed = RawDocument("changed text", "guide.md", "markdown")

        first = index_documents(
            [original],
            embedding_provider=self.embedder,
            vector_store=self.store,
            tokenizer=self.tokenizer,
        )
        second = index_documents(
            [original],
            embedding_provider=self.embedder,
            vector_store=self.store,
            tokenizer=self.tokenizer,
        )
        third = index_documents(
            [changed],
            embedding_provider=self.embedder,
            vector_store=self.store,
            tokenizer=self.tokenizer,
        )

        self.assertEqual(first[0].id, second[0].id)
        self.assertNotEqual(first[0].id, third[0].id)

    def test_embedding_count_mismatch_is_rejected_before_storage(self) -> None:
        embedder = FakeEmbeddingProvider(return_too_few=True)
        source = RawDocument("some content", "guide.md", "markdown")

        with self.assertRaisesRegex(ValueError, "embedding"):
            index_documents(
                [source],
                embedding_provider=embedder,
                vector_store=self.store,
                tokenizer=self.tokenizer,
            )

        self.assertEqual(self.store.upsert_calls, [])

    def test_empty_input_does_not_call_external_components(self) -> None:
        records = index_documents(
            [],
            embedding_provider=self.embedder,
            vector_store=self.store,
            tokenizer=self.tokenizer,
        )

        self.assertEqual(records, [])
        self.assertEqual(self.embedder.document_calls, [])
        self.assertEqual(self.store.upsert_calls, [])


class AdapterTests(unittest.TestCase):
    def test_chroma_upsert_uses_columnar_record_payload(self) -> None:
        client = FakeChromaClient()
        store = ChromaVectorStore(client=client, collection_name="rag_documents")
        embedder = FakeEmbeddingProvider()
        records = index_documents(
            [RawDocument("evidence", "guide.md", "markdown")],
            embedding_provider=embedder,
            vector_store=store,
            tokenizer=WhitespaceTokenizer(),
        )

        self.assertEqual(
            client.collection_options,
            {"name": "rag_documents", "embedding_function": None},
        )
        self.assertEqual(client.collection.payload["ids"], [records[0].id])
        self.assertEqual(client.collection.payload["documents"], ["evidence"])
        self.assertEqual(client.collection.payload["embeddings"], [[1.0, 0.0]])
        self.assertEqual(
            client.collection.payload["metadatas"],
            [records[0].metadata],
        )

    def test_chroma_delete_uses_an_explicit_id_list(self) -> None:
        client = FakeChromaClient()
        store = ChromaVectorStore(client=client, collection_name="rag_documents")

        store.delete(["chunk-1", "chunk-2"])

        self.assertEqual(client.collection.deleted_ids, ["chunk-1", "chunk-2"])

    def test_sentence_transformer_uses_retrieval_specific_normalized_encoding(self) -> None:
        model = FakeSentenceTransformer()
        provider = SentenceTransformerEmbeddingProvider(model=model)

        documents = provider.embed_documents(["first", "second"])
        query = provider.embed_query("question")

        self.assertEqual(documents, [[0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(query, [8.0, 1.0])
        expected_options = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        self.assertEqual(model.document_options, expected_options)
        self.assertEqual(model.query_options, expected_options)


if __name__ == "__main__":
    unittest.main()
