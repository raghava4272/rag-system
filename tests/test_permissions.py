"""Document permission configuration and exact-source authorization tests."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src.permissions import DocumentPermissions


class DocumentPermissionsTests(unittest.TestCase):
    def test_maps_users_to_immutable_exact_source_sets(self) -> None:
        permissions = DocumentPermissions(
            {
                "alice": ["data/raw/public.md", "data/raw/alice/private.pdf"],
                "bob": [],
            }
        )

        self.assertEqual(
            permissions.allowed_sources("alice"),
            frozenset({"data/raw/public.md", "data/raw/alice/private.pdf"}),
        )
        self.assertEqual(permissions.allowed_sources("bob"), frozenset())
        self.assertIsNone(permissions.allowed_sources("unknown-user"))
        self.assertNotIn("private.pdf", repr(permissions))

    def test_json_configuration_is_deny_by_default_and_rejects_ambiguous_rules(self) -> None:
        invalid_values = (
            "",
            "[]",
            "not-json",
            json.dumps({"bad user": ["data/raw/guide.md"]}),
            json.dumps({"alice": "data/raw/guide.md"}),
            json.dumps({"alice": [""]}),
            json.dumps({"alice": ["data/raw/guide.md", "data/raw/guide.md"]}),
            json.dumps({"alice": ["*"]}),
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    DocumentPermissions.from_json(value)

    def test_environment_loader_requires_explicit_runtime_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                DocumentPermissions.from_environment()

        encoded = json.dumps({"local-user": ["data/raw/guide.md"]})
        with patch.dict(
            os.environ,
            {"RAG_DOCUMENT_PERMISSIONS": encoded},
            clear=True,
        ):
            permissions = DocumentPermissions.from_environment()

        self.assertEqual(
            permissions.allowed_sources("local-user"),
            frozenset({"data/raw/guide.md"}),
        )


if __name__ == "__main__":
    unittest.main()
