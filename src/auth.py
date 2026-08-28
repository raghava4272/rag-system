"""Small, provider-neutral bearer authentication primitives."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping


API_KEY_HASHES_ENV = "RAG_API_KEY_HASHES"
MIN_API_KEY_LENGTH = 32
MAX_API_KEY_LENGTH = 512
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def hash_api_key(api_key: str) -> str:
    """Return the lowercase SHA-256 digest stored in runtime configuration."""

    if not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if not MIN_API_KEY_LENGTH <= len(api_key) <= MAX_API_KEY_LENGTH:
        raise ValueError(
            f"api_key must contain {MIN_API_KEY_LENGTH}-{MAX_API_KEY_LENGTH} "
            "characters"
        )
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class APIKeyAuthenticator:
    """Map bearer keys to trusted user IDs using constant-time hash checks."""

    def __init__(self, key_hashes: Mapping[str, str]) -> None:
        if not key_hashes:
            raise ValueError("At least one API key hash must be configured")

        credentials: list[tuple[str, str]] = []
        seen_hashes: set[str] = set()
        for user_id, digest in key_hashes.items():
            if not isinstance(user_id, str) or not USER_ID_PATTERN.fullmatch(user_id):
                raise ValueError(
                    "User IDs must start with a letter or number and contain only "
                    "letters, numbers, dots, underscores, or hyphens"
                )
            if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
                raise ValueError(
                    "API key values must be lowercase SHA-256 hashes, not raw keys"
                )
            if digest in seen_hashes:
                raise ValueError("Each API key hash must belong to exactly one user")
            seen_hashes.add(digest)
            credentials.append((user_id, digest))

        self._credentials = tuple(credentials)

    def __repr__(self) -> str:
        return f"APIKeyAuthenticator(user_count={len(self._credentials)})"

    @classmethod
    def from_json(cls, encoded: str) -> APIKeyAuthenticator:
        if not isinstance(encoded, str) or not encoded.strip():
            raise ValueError("API key hash configuration cannot be empty")
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError("API key hash configuration must be valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("API key hash configuration must be a JSON object")
        return cls(value)

    @classmethod
    def from_environment(cls) -> APIKeyAuthenticator:
        encoded = os.getenv(API_KEY_HASHES_ENV)
        if not encoded:
            raise RuntimeError(
                f"{API_KEY_HASHES_ENV} is not configured with user-to-hash mappings"
            )
        try:
            return cls.from_json(encoded)
        except ValueError as error:
            raise RuntimeError(f"{API_KEY_HASHES_ENV} is invalid") from error

    def authenticate(self, api_key: str) -> str | None:
        if not isinstance(api_key, str):
            return None
        if not MIN_API_KEY_LENGTH <= len(api_key) <= MAX_API_KEY_LENGTH:
            return None
        try:
            presented_digest = hash_api_key(api_key)
        except (TypeError, UnicodeEncodeError, ValueError):
            return None

        matched_user: str | None = None
        for user_id, configured_digest in self._credentials:
            if hmac.compare_digest(presented_digest, configured_digest):
                matched_user = user_id
        return matched_user
