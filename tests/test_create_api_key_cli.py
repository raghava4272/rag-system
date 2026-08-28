"""Tests for the one-time local API credential generator."""

from __future__ import annotations

import io
import json
import unittest

from scripts.create_api_key import main
from src.auth import hash_api_key


class CreateAPIKeyCliTests(unittest.TestCase):
    def test_prints_a_generated_key_and_only_its_hash_in_configuration(self) -> None:
        output = io.StringIO()
        generated_key = "generated-api-key-with-at-least-thirty-two-characters"

        exit_code = main(
            ["--user-id", "local-user"],
            token_factory=lambda: generated_key,
            stdout=output,
        )

        self.assertEqual(exit_code, 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[1], generated_key)
        encoded_configuration = lines[-1].removeprefix("RAG_API_KEY_HASHES=")
        self.assertEqual(
            json.loads(encoded_configuration),
            {"local-user": hash_api_key(generated_key)},
        )
        self.assertNotIn(generated_key, encoded_configuration)

    def test_invalid_user_id_is_rejected_before_generating_a_key(self) -> None:
        calls: list[None] = []

        with self.assertRaises(ValueError):
            main(
                ["--user-id", "not a safe user"],
                token_factory=lambda: calls.append(None) or "unused-key-value",
                stdout=io.StringIO(),
            )

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
