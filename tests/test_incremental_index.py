"""Behavioral tests for manifest-backed incremental indexing."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from src.embed import EmbeddedChunk, StoredChunk
from src.index_manifest import IndexLockError, acquire_index_lock
from src.pipeline import build_index


class WhitespaceTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class RecordingEmbeddingProvider:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.document_calls.append(batch)
        return [[float(len(text.split())), 1.0] for text in batch]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0]


class StatefulVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, EmbeddedChunk] = {}
        self.upsert_calls: list[list[EmbeddedChunk]] = []
        self.delete_calls: list[list[str]] = []

    def upsert(self, records: Sequence[EmbeddedChunk]) -> None:
        batch = list(records)
        self.upsert_calls.append(batch)
        self.records.update({record.id: record for record in batch})

    def delete(self, record_ids: Sequence[str]) -> None:
        ids = list(record_ids)
        self.delete_calls.append(ids)
        for record_id in ids:
            self.records.pop(record_id, None)

    def all_chunks(self) -> list[StoredChunk]:
        return [
            StoredChunk(record.id, record.text, dict(record.metadata))
            for record in self.records.values()
        ]


class IncrementalIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.sources = self.root / "raw"
        self.sources.mkdir()
        self.manifest = self.root / "chroma" / "index-manifest.json"
        self.tokenizer = WhitespaceTokenizer()
        self.embedder = RecordingEmbeddingProvider()
        self.store = StatefulVectorStore()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def build(self, *, embedding_model: str = "test-embedding-v1"):
        return build_index(
            self.sources,
            embedding_provider=self.embedder,
            vector_store=self.store,
            tokenizer=self.tokenizer,
            chunk_size=4,
            overlap=1,
            manifest_path=self.manifest,
            embedding_model=embedding_model,
        )

    def test_second_identical_build_skips_all_embedding_and_storage(self) -> None:
        (self.sources / "a.md").write_text("one two three four five", encoding="utf-8")
        (self.sources / "b.md").write_text("six seven", encoding="utf-8")

        first = self.build()
        self.embedder.document_calls.clear()
        self.store.upsert_calls.clear()
        self.store.delete_calls.clear()
        second = self.build()

        self.assertTrue(self.manifest.exists())
        self.assertEqual(first.indexed_chunk_count, 3)
        self.assertEqual(second.indexed_chunk_count, 0)
        self.assertEqual(second.unchanged_source_file_count, 2)
        self.assertEqual(second.total_chunk_count, 3)
        self.assertEqual(self.embedder.document_calls, [])
        self.assertEqual(self.store.upsert_calls, [])
        self.assertEqual(self.store.delete_calls, [])

    def test_changed_source_is_reembedded_without_touching_unchanged_source(self) -> None:
        changed = self.sources / "changed.md"
        unchanged = self.sources / "unchanged.md"
        changed.write_text("old source text", encoding="utf-8")
        unchanged.write_text("stable evidence", encoding="utf-8")
        self.build()
        old_ids_by_source = {
            source: {record.id for record in self.store.records.values() if record.metadata["source"] == source}
            for source in (str(changed), str(unchanged))
        }

        self.embedder.document_calls.clear()
        changed.write_text("new source text with additions", encoding="utf-8")
        result = self.build()

        self.assertEqual(result.changed_source_file_count, 1)
        self.assertEqual(result.unchanged_source_file_count, 1)
        self.assertEqual(len(self.embedder.document_calls), 1)
        embedded_text = " ".join(self.embedder.document_calls[0])
        self.assertIn("new source", embedded_text)
        self.assertNotIn("stable evidence", embedded_text)
        remaining_unchanged_ids = {
            record.id
            for record in self.store.records.values()
            if record.metadata["source"] == str(unchanged)
        }
        self.assertEqual(remaining_unchanged_ids, old_ids_by_source[str(unchanged)])
        self.assertTrue(old_ids_by_source[str(changed)].isdisjoint(self.store.records))

    def test_deleted_source_records_are_removed_without_embedding(self) -> None:
        removed = self.sources / "removed.md"
        kept = self.sources / "kept.md"
        removed.write_text("remove this evidence", encoding="utf-8")
        kept.write_text("keep this evidence", encoding="utf-8")
        self.build()
        removed_ids = {
            record.id
            for record in self.store.records.values()
            if record.metadata["source"] == str(removed)
        }

        self.embedder.document_calls.clear()
        removed.unlink()
        result = self.build()

        self.assertEqual(result.removed_source_file_count, 1)
        self.assertEqual(result.indexed_chunk_count, 0)
        self.assertEqual(self.embedder.document_calls, [])
        self.assertTrue(removed_ids.isdisjoint(self.store.records))
        self.assertEqual(
            {record.metadata["source"] for record in self.store.records.values()},
            {str(kept)},
        )

    def test_configuration_change_rebuilds_every_source(self) -> None:
        (self.sources / "a.md").write_text("one two", encoding="utf-8")
        (self.sources / "b.md").write_text("three four", encoding="utf-8")
        self.build()

        self.embedder.document_calls.clear()
        result = self.build(embedding_model="test-embedding-v2")

        self.assertEqual(result.changed_source_file_count, 2)
        self.assertEqual(result.unchanged_source_file_count, 0)
        self.assertEqual(len(self.embedder.document_calls), 1)

    def test_missing_and_orphaned_chroma_records_are_repaired(self) -> None:
        source = self.sources / "guide.md"
        source.write_text("one two three four five", encoding="utf-8")
        self.build()
        missing_id = next(iter(self.store.records))
        self.store.records.pop(missing_id)
        orphan = EmbeddedChunk(
            id="orphan",
            text="stale",
            embedding=[0.0],
            metadata={"source": "removed.md"},
        )
        self.store.records[orphan.id] = orphan

        self.embedder.document_calls.clear()
        result = self.build()

        self.assertEqual(result.changed_source_file_count, 1)
        self.assertEqual(len(self.embedder.document_calls), 1)
        self.assertNotIn("orphan", self.store.records)
        self.assertEqual(result.total_chunk_count, len(self.store.records))

    def test_invalid_manifest_fails_without_mutating_the_collection(self) -> None:
        source = self.sources / "guide.md"
        source.write_text("trusted evidence", encoding="utf-8")
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text("not-json", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "manifest"):
            self.build()

        self.assertEqual(self.embedder.document_calls, [])
        self.assertEqual(self.store.records, {})

    def test_concurrent_build_fails_before_embedding_or_storage_mutation(self) -> None:
        (self.sources / "guide.md").write_text("trusted evidence", encoding="utf-8")

        with acquire_index_lock(self.manifest):
            with self.assertRaisesRegex(IndexLockError, "already running"):
                self.build()

        self.assertEqual(self.embedder.document_calls, [])
        self.assertEqual(self.store.upsert_calls, [])
        self.assertEqual(self.store.delete_calls, [])
        self.assertEqual(self.store.records, {})


if __name__ == "__main__":
    unittest.main()
