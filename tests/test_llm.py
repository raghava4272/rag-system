"""Behavioral tests for the Groq language-model adapter."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.llm import DEFAULT_GROQ_MODEL, GROQ_BASE_URL, GroqLanguageModel


class FakeResponses:
    def __init__(self, output_text: object) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **options: object) -> SimpleNamespace:
        self.calls.append(options)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text: object) -> None:
        self.responses = FakeResponses(output_text)


class GroqLanguageModelTests(unittest.TestCase):
    def test_generate_uses_groq_responses_api_and_default_model(self) -> None:
        client = FakeClient("Grounded answer [1].")
        model = GroqLanguageModel(client=client)

        result = model.generate("grounded prompt")

        self.assertEqual(result, "Grounded answer [1].")
        self.assertEqual(
            client.responses.calls,
            [{"model": DEFAULT_GROQ_MODEL, "input": "grounded prompt"}],
        )

    def test_model_name_is_configurable(self) -> None:
        client = FakeClient("Answer [1].")
        model = GroqLanguageModel(
            model="openai/gpt-oss-120b",
            client=client,
        )

        model.generate("prompt")

        self.assertEqual(
            client.responses.calls[0]["model"],
            "openai/gpt-oss-120b",
        )

    def test_blank_provider_response_is_rejected(self) -> None:
        model = GroqLanguageModel(client=FakeClient("  \n"))

        with self.assertRaisesRegex(ValueError, "empty"):
            model.generate("prompt")

    def test_missing_api_key_is_rejected_before_client_creation(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                GroqLanguageModel()

    def test_public_defaults_are_explicit(self) -> None:
        self.assertEqual(DEFAULT_GROQ_MODEL, "openai/gpt-oss-20b")
        self.assertEqual(GROQ_BASE_URL, "https://api.groq.com/openai/v1")


if __name__ == "__main__":
    unittest.main()
