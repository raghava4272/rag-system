"""Authentication configuration and credential verification tests."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src.auth import APIKeyAuthenticator, hash_api_key


TEST_KEY = "test-api-key-with-at-least-thirty-two-characters"


class APIKeyAuthenticatorTests(unittest.TestCase):
    def test_valid_key_maps_to_its_server_configured_user(self) -> None:
        authenticator = APIKeyAuthenticator(
            {"user-123": hash_api_key(TEST_KEY)},
        )

        self.assertEqual(authenticator.authenticate(TEST_KEY), "user-123")
        self.assertIsNone(authenticator.authenticate("wrong-api-key-value"))
        self.assertIsNone(authenticator.authenticate(""))

    def test_json_configuration_requires_user_to_hash_mapping(self) -> None:
        configured = APIKeyAuthenticator.from_json(
            json.dumps({"alice": hash_api_key(TEST_KEY)})
        )

        self.assertEqual(configured.authenticate(TEST_KEY), "alice")

        invalid_values = (
            "",
            "[]",
            "not-json",
            json.dumps({"bad user id": hash_api_key(TEST_KEY)}),
            json.dumps({"alice": TEST_KEY}),
            json.dumps(
                {
                    "alice": hash_api_key(TEST_KEY),
                    "bob": hash_api_key(TEST_KEY),
                }
            ),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    APIKeyAuthenticator.from_json(value)

    def test_environment_loader_requires_explicit_runtime_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                APIKeyAuthenticator.from_environment()

        encoded = json.dumps({"local-user": hash_api_key(TEST_KEY)})
        with patch.dict(os.environ, {"RAG_API_KEY_HASHES": encoded}, clear=True):
            authenticator = APIKeyAuthenticator.from_environment()

        self.assertEqual(authenticator.authenticate(TEST_KEY), "local-user")

    def test_hash_is_stable_lowercase_sha256_without_retaining_the_key(self) -> None:
        digest = hash_api_key(TEST_KEY)
        authenticator = APIKeyAuthenticator({"alice": digest})

        self.assertEqual(len(digest), 64)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertNotIn(TEST_KEY, repr(authenticator))


if __name__ == "__main__":
    unittest.main()
