"""Behavioral tests for local document ingestion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ingest import load_document, load_documents


class FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class FakePdfReader:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages


class IngestionTests(unittest.TestCase):
    def test_markdown_preserves_content_and_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "guide.md"
            content = "# Retrieval\n\nUse focused, attributable context.\n"
            path.write_text(content, encoding="utf-8")

            documents = load_document(path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, content)
        self.assertEqual(documents[0].source, str(path))
        self.assertEqual(documents[0].document_type, "markdown")
        self.assertIsNone(documents[0].page_number)

    def test_pdf_emits_one_document_per_nonempty_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "paper.pdf"
            path.touch()
            reader = FakePdfReader(
                [FakePage("First page"), FakePage("  \n"), FakePage("Third page")]
            )

            documents = load_document(path, pdf_reader_factory=lambda _: reader)

        self.assertEqual([document.text for document in documents], ["First page", "Third page"])
        self.assertEqual([document.page_number for document in documents], [1, 3])
        self.assertTrue(all(document.source == str(path) for document in documents))
        self.assertTrue(
            all(document.document_type == "pdf" for document in documents)
        )

    def test_directory_loading_is_recursive_ordered_and_filters_file_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (root / "b.md").write_text("B", encoding="utf-8")
            (nested / "a.mdx").write_text("A", encoding="utf-8")
            (root / "ignored.txt").write_text("ignored", encoding="utf-8")

            documents = load_documents(root)

        self.assertEqual([document.text for document in documents], ["B", "A"])
        self.assertEqual(
            [Path(document.source).name for document in documents],
            ["b.md", "a.mdx"],
        )

    def test_unsupported_single_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notes.txt"
            path.write_text("unsupported", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported document type"):
                load_document(path)

    def test_missing_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.md"

            with self.assertRaises(FileNotFoundError):
                load_document(missing_path)


if __name__ == "__main__":
    unittest.main()
