"""Behavioral tests for the token-based document chunker."""

from __future__ import annotations

import unittest

from src.chunk import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_text


class WhitespaceTokenizer:
    """A deterministic tokenizer used to keep the unit tests dependency-free."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def make_document(token_count: int) -> str:
    return " ".join(f"token-{index}" for index in range(token_count))


class ChunkTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()

    def test_long_document_uses_expected_size_and_overlap(self) -> None:
        chunks = chunk_text(make_document(1_500), tokenizer=self.tokenizer)

        self.assertEqual([chunk.token_count for chunk in chunks], [700, 700, 300])
        self.assertTrue(
            all(500 <= chunk.token_count <= 800 for chunk in chunks[:-1]),
            "Every full chunk should stay within the documented 500-800 token range",
        )

        for previous, current in zip(chunks, chunks[1:]):
            previous_tokens = self.tokenizer.encode(previous.text)
            current_tokens = self.tokenizer.encode(current.text)
            self.assertEqual(
                previous_tokens[-DEFAULT_OVERLAP:],
                current_tokens[:DEFAULT_OVERLAP],
            )
            self.assertEqual(
                current.token_start,
                previous.token_end - DEFAULT_OVERLAP,
            )

    def test_chunk_metadata_is_stable_and_ordered(self) -> None:
        chunks = chunk_text(make_document(1_500), tokenizer=self.tokenizer)

        self.assertEqual([chunk.index for chunk in chunks], [0, 1, 2])
        self.assertEqual(
            [(chunk.token_start, chunk.token_end) for chunk in chunks],
            [(0, 700), (600, 1_300), (1_200, 1_500)],
        )
        self.assertTrue(
            all(
                chunk.token_count == chunk.token_end - chunk.token_start
                for chunk in chunks
            )
        )

    def test_short_document_is_returned_as_one_chunk(self) -> None:
        text = "A short document remains a single chunk."

        chunks = chunk_text(text, tokenizer=self.tokenizer)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].index, 0)
        self.assertEqual(chunks[0].token_start, 0)
        self.assertEqual(chunks[0].token_end, 7)

    def test_empty_document_produces_no_chunks(self) -> None:
        for text in ("", "   ", "\n\t"):
            with self.subTest(text=text):
                self.assertEqual(chunk_text(text, tokenizer=self.tokenizer), [])

    def test_invalid_chunk_settings_are_rejected(self) -> None:
        invalid_settings = (
            (0, 0),
            (-1, 0),
            (100, -1),
            (100, 100),
            (100, 101),
        )
        for chunk_size, overlap in invalid_settings:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_text(
                        "some text",
                        chunk_size=chunk_size,
                        overlap=overlap,
                        tokenizer=self.tokenizer,
                    )

    def test_documented_defaults_are_explicit(self) -> None:
        self.assertEqual(DEFAULT_CHUNK_SIZE, 700)
        self.assertEqual(DEFAULT_OVERLAP, 100)


if __name__ == "__main__":
    unittest.main()
