"""Tests for the container's stable command interface."""

from __future__ import annotations

import io
import unittest

from scripts.container import main


class ContainerCommandTests(unittest.TestCase):
    def test_help_lists_every_supported_job(self) -> None:
        output = io.StringIO()

        exit_code = main(["--help"], commands={}, stdout=output)

        self.assertEqual(exit_code, 0)
        self.assertIn("index", output.getvalue())
        self.assertIn("ask", output.getvalue())
        self.assertIn("evaluate", output.getvalue())

    def test_dispatches_remaining_arguments_and_preserves_exit_code(self) -> None:
        calls: list[list[str]] = []

        def ask(arguments: list[str]) -> int:
            calls.append(arguments)
            return 1

        exit_code = main(
            ["ask", "What is RAG?", "--top-k", "3"],
            commands={"ask": ask},
            stdout=io.StringIO(),
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(calls, [["What is RAG?", "--top-k", "3"]])

    def test_unknown_command_is_a_usage_error(self) -> None:
        error = io.StringIO()

        exit_code = main(
            ["serve"],
            commands={},
            stdout=io.StringIO(),
            stderr=error,
        )

        self.assertEqual(exit_code, 2)
        self.assertIn("Unknown command: serve", error.getvalue())


if __name__ == "__main__":
    unittest.main()
