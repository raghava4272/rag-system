"""Token-based text chunking with deterministic overlap metadata."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, TypeVar, cast


DEFAULT_CHUNK_SIZE = 700
DEFAULT_OVERLAP = 100
DEFAULT_ENCODING = "cl100k_base"

Token = TypeVar("Token")


class Tokenizer(Protocol[Token]):
    """Minimal tokenizer interface required by :func:`chunk_text`."""

    def encode(self, text: str) -> list[Token]: ...

    def decode(self, tokens: list[Token]) -> str: ...


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk plus its position in the source document's token stream."""

    text: str
    index: int
    token_start: int
    token_end: int

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@lru_cache(maxsize=1)
def _default_tokenizer() -> Tokenizer[int]:
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError(
            "The default tokenizer requires tiktoken. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`, or pass a tokenizer "
            "implementing encode() and decode()."
        ) from error

    return cast(Tokenizer[int], tiktoken.get_encoding(DEFAULT_ENCODING))


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    tokenizer: Tokenizer[Token] | None = None,
) -> list[TextChunk]:
    """Split ``text`` into token windows and attach stable position metadata.

    Full chunks contain ``chunk_size`` tokens. Consecutive chunks repeat exactly
    ``overlap`` tokens, while the final chunk may be shorter. Supplying a
    tokenizer is useful for matching the token accounting of a specific model.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text.strip():
        return []

    active_tokenizer = tokenizer
    if active_tokenizer is None:
        active_tokenizer = cast(Tokenizer[Token], _default_tokenizer())

    tokens = active_tokenizer.encode(text)
    if not tokens:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - overlap

    for token_start in range(0, len(tokens), step):
        token_end = min(token_start + chunk_size, len(tokens))
        chunks.append(
            TextChunk(
                text=active_tokenizer.decode(tokens[token_start:token_end]),
                index=len(chunks),
                token_start=token_start,
                token_end=token_end,
            )
        )
        if token_end == len(tokens):
            break

    return chunks
