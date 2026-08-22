"""Tests for claim-unit citation coverage in generated answers."""

from __future__ import annotations

import unittest

from src.embed import RetrievedChunk
from src.generate import InvalidCitationError, generate_answer


class FakeLanguageModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def source_chunk(record_id: str, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=record_id,
        text=f"Source evidence from {record_id}.",
        metadata={"source": f"{record_id}.md", "chunk_index": 0},
        distance=0.1 * rank,
        rank=rank,
    )


class CitationCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            source_chunk("retrieval", rank=1),
            source_chunk("citations", rank=2),
        ]

    def test_uncited_paragraph_is_rejected_even_when_another_paragraph_is_cited(self) -> None:
        model = FakeLanguageModel(
            "RAG retrieves evidence before generation [1].\n\n"
            "It always guarantees a correct answer."
        )

        with self.assertRaisesRegex(InvalidCitationError, "coverage"):
            generate_answer("Explain RAG", self.chunks, language_model=model)

    def test_every_list_item_requires_its_own_citation(self) -> None:
        model = FakeLanguageModel(
            "RAG has two useful properties [1].\n\n"
            "- Retrieval supplies source context [1].\n"
            "- Generation can never hallucinate."
        )

        with self.assertRaisesRegex(InvalidCitationError, "coverage"):
            generate_answer("Explain RAG", self.chunks, language_model=model)

    def test_citation_on_heading_does_not_cover_following_paragraph(self) -> None:
        model = FakeLanguageModel(
            "## Retrieval overview [1]\n\n"
            "This paragraph makes an unsupported factual claim."
        )

        with self.assertRaisesRegex(InvalidCitationError, "coverage"):
            generate_answer("Explain retrieval", self.chunks, language_model=model)

    def test_headings_paragraphs_and_bullets_with_citations_are_accepted(self) -> None:
        model = FakeLanguageModel(
            "## Summary\n\n"
            "RAG retrieves evidence before generation [1].\n\n"
            "- Citations connect claims to sources [2].\n"
            "- Retrieved chunks provide context [1] [2]."
        )

        answer = generate_answer("Explain RAG", self.chunks, language_model=model)

        self.assertEqual(answer.text, model.response)
        self.assertEqual([citation.number for citation in answer.citations], [1, 2])

    def test_wrapped_lines_form_one_paragraph_and_may_share_a_citation(self) -> None:
        model = FakeLanguageModel(
            "RAG first retrieves relevant evidence\n"
            "and then supplies it to the generator [1]."
        )

        answer = generate_answer("Explain RAG", self.chunks, language_model=model)

        self.assertEqual(answer.text, model.response)

    def test_prompt_requires_citations_for_each_claim_unit(self) -> None:
        model = FakeLanguageModel("Grounded answer [1].")

        generate_answer("Explain RAG", self.chunks, language_model=model)

        self.assertIn(
            "Every factual paragraph and list item",
            model.prompts[0],
        )


if __name__ == "__main__":
    unittest.main()
