#!/usr/bin/env python3
"""Stable command dispatcher for the one-shot RAG container image."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


Command = Callable[[list[str]], int]

HELP = """usage: rag-system <command> [arguments]

Run the containerized RAG pipeline as a one-shot job.

commands:
  index       build or update the persistent Chroma index
  ask         answer one question using the indexed corpus
  evaluate    run the Phase 3 golden evaluation and quality gate

examples:
  rag-system index
  rag-system ask "What is reciprocal rank fusion?"
  rag-system evaluate --threshold 0.8
"""


def _default_commands() -> dict[str, Command]:
    from eval.run_eval import main as evaluate
    from scripts.ask import main as ask
    from scripts.build_index import main as index

    return {
        "index": index,
        "ask": ask,
        "evaluate": evaluate,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    commands: dict[str, Command] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(HELP, file=stdout, end="")
        return 0

    command_name, *command_arguments = arguments
    resolved_commands = _default_commands() if commands is None else commands
    command = resolved_commands.get(command_name)
    if command is None:
        print(f"Unknown command: {command_name}", file=stderr)
        print("Run with --help to list supported commands.", file=stderr)
        return 2

    return command(command_arguments)


def cli() -> int:
    try:
        return main()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
