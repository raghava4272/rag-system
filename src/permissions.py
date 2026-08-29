"""Deny-by-default document permissions for authenticated API users."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

from src.auth import USER_ID_PATTERN


DOCUMENT_PERMISSIONS_ENV = "RAG_DOCUMENT_PERMISSIONS"
MAX_SOURCE_LENGTH = 2_048


class DocumentPermissions:
    """Map trusted user IDs to exact indexed source identifiers."""

    def __init__(self, permissions: Mapping[str, Sequence[str]]) -> None:
        if not permissions:
            raise ValueError("At least one document permission rule must be configured")

        rules: dict[str, frozenset[str]] = {}
        for user_id, sources in permissions.items():
            if not isinstance(user_id, str) or not USER_ID_PATTERN.fullmatch(user_id):
                raise ValueError(
                    "Permission user IDs must match authenticated user ID syntax"
                )
            if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
                raise ValueError("Each permission rule must be a JSON array of sources")

            validated_sources: list[str] = []
            for source in sources:
                if (
                    not isinstance(source, str)
                    or not source
                    or source != source.strip()
                    or len(source) > MAX_SOURCE_LENGTH
                    or any(character in source for character in ("\n", "\r", "\0"))
                ):
                    raise ValueError("Document sources must be nonempty exact strings")
                if source == "*":
                    raise ValueError("Wildcard document access is not supported")
                validated_sources.append(source)

            if len(validated_sources) != len(set(validated_sources)):
                raise ValueError("A permission rule cannot contain duplicate sources")
            rules[user_id] = frozenset(validated_sources)

        self._rules = rules

    def __repr__(self) -> str:
        source_count = sum(len(sources) for sources in self._rules.values())
        return (
            f"DocumentPermissions(user_count={len(self._rules)}, "
            f"source_count={source_count})"
        )

    @classmethod
    def from_json(cls, encoded: str) -> DocumentPermissions:
        if not isinstance(encoded, str) or not encoded.strip():
            raise ValueError("Document permission configuration cannot be empty")
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError("Document permission configuration must be valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Document permission configuration must be a JSON object")
        return cls(value)

    @classmethod
    def from_environment(cls) -> DocumentPermissions:
        encoded = os.getenv(DOCUMENT_PERMISSIONS_ENV)
        if not encoded:
            raise RuntimeError(
                f"{DOCUMENT_PERMISSIONS_ENV} is not configured with user permissions"
            )
        try:
            return cls.from_json(encoded)
        except ValueError as error:
            raise RuntimeError(f"{DOCUMENT_PERMISSIONS_ENV} is invalid") from error

    def allowed_sources(self, user_id: str) -> frozenset[str] | None:
        """Return an immutable exact-source scope, or None when no rule exists."""

        return self._rules.get(user_id)
