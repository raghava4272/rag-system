"""Language-model provider adapters."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqLanguageModel:
    """Generate text through Groq's OpenAI-compatible Responses API."""

    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is None:
            resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
            if not resolved_api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not configured. Copy .env.example to .env "
                    "and add a key from https://console.groq.com/keys."
                )
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "The Groq adapter requires the openai client package. Install "
                    "dependencies with `python3 -m pip install -r requirements.txt`."
                ) from error
            client = OpenAI(
                api_key=resolved_api_key,
                base_url=GROQ_BASE_URL,
            )
        self.client = client

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("Groq returned an empty text response")
        return output_text.strip()

