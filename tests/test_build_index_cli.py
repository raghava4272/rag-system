"""Tests for incremental indexing CLI wiring and reporting."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import build_index as build_index_cli
from src.index_manifest import default_manifest_path
from src.pipeline import IndexBuildResult


class BuildIndexCliTests(unittest.TestCase):
    def test_cli_uses_collection_specific_manifest_and_reports_reused_work(self) -> None:
        result = IndexBuildResult(
            source_file_count=28,
            raw_document_count=78,
            indexed_chunk_count=0,
            changed_source_file_count=0,
            unchanged_source_file_count=28,
            removed_source_file_count=0,
            total_chunk_count=235,
        )
        output = io.StringIO()
        chroma_path = Path("custom/chroma")

        with (
            patch.object(build_index_cli, "SentenceTransformerEmbeddingProvider"),
            patch.object(build_index_cli, "ChromaVectorStore"),
            patch.object(build_index_cli, "build_index", return_value=result) as build,
            redirect_stdout(output),
        ):
            exit_code = build_index_cli.main(
                [
                    "data/raw",
                    "--chroma-path",
                    str(chroma_path),
                    "--collection",
                    "docs-v2",
                    "--embedding-model",
                    "embedding-v2",
                    "--lock-timeout-seconds",
                    "2.5",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            build.call_args.kwargs["manifest_path"],
            default_manifest_path(chroma_path, "docs-v2"),
        )
        self.assertEqual(build.call_args.kwargs["embedding_model"], "embedding-v2")
        self.assertEqual(build.call_args.kwargs["lock_timeout_seconds"], 2.5)
        self.assertIn("Embedded 0 chunks from 0 changed source files", output.getvalue())
        self.assertIn("reused 28 unchanged source files", output.getvalue())
        self.assertIn("235 total chunks", output.getvalue())

    def test_cli_rejects_an_invalid_lock_timeout(self) -> None:
        for timeout in ("-0.1", "nan", "inf"):
            with self.subTest(timeout=timeout):
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    build_index_cli.build_parser().parse_args(
                        ["--lock-timeout-seconds", timeout]
                    )

                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
