"""Tests for formatting answers returned by the ask CLI."""

from __future__ import annotations

import unittest

from scripts.ask import format_answer
from src.generate import Citation, GeneratedAnswer


class AskCliFormattingTests(unittest.TestCase):
    def test_answer_includes_numbered_pdf_and_markdown_sources(self) -> None:
        answer = GeneratedAnswer(
            text="Retrieved evidence supports the answer [1] [2].",
            citations=[
                Citation(
                    number=1,
                    record_id="pdf-chunk",
                    source="data/raw/paper.pdf",
                    page_number=4,
                    chunk_index=0,
                ),
                Citation(
                    number=2,
                    record_id="md-chunk",
                    source="data/raw/guide.md",
                    page_number=None,
                    chunk_index=3,
                ),
            ],
        )

        output = format_answer(answer)

        self.assertIn(answer.text, output)
        self.assertIn("Sources:", output)
        self.assertIn("[1] data/raw/paper.pdf (page 4, chunk 0)", output)
        self.assertIn("[2] data/raw/guide.md (chunk 3)", output)

    def test_answer_without_citations_omits_sources_section(self) -> None:
        answer = GeneratedAnswer(text="No relevant context.", citations=[])

        self.assertEqual(format_answer(answer), "No relevant context.")


if __name__ == "__main__":
    unittest.main()
