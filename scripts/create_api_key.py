#!/usr/bin/env python3
"""Generate one strong bearer key and its safe runtime configuration hash."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auth import APIKeyAuthenticator, hash_api_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a RAG API bearer key for one user."
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Stable user ID containing letters, numbers, dots, underscores, or hyphens",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    stdout: TextIO = sys.stdout,
) -> int:
    args = build_parser().parse_args(argv)

    # Validate the identity before generating and displaying a one-time secret.
    APIKeyAuthenticator({args.user_id: "0" * 64})
    api_key = token_factory()
    digest = hash_api_key(api_key)
    encoded = json.dumps(
        {args.user_id: digest},
        separators=(",", ":"),
        sort_keys=True,
    )

    print("API key (save it now; it will not be shown again):", file=stdout)
    print(api_key, file=stdout)
    print(file=stdout)
    print("Add this line to .env:", file=stdout)
    print(f"RAG_API_KEY_HASHES={encoded}", file=stdout)
    return 0


def cli() -> int:
    try:
        return main()
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
