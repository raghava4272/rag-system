"""Tests for the process-held incremental index lock."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.index_manifest import IndexLockError, acquire_index_lock, index_lock_path


class IndexLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.manifest = Path(self.temporary_directory.name) / "chroma" / "manifest.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_second_writer_fails_while_the_first_writer_holds_the_lock(self) -> None:
        with acquire_index_lock(self.manifest):
            with self.assertRaisesRegex(IndexLockError, "already running"):
                with acquire_index_lock(self.manifest, timeout_seconds=0):
                    self.fail("a second writer acquired the same index lock")

    def test_lock_is_available_again_after_the_holder_releases_it(self) -> None:
        with acquire_index_lock(self.manifest):
            pass

        with acquire_index_lock(self.manifest):
            pass

    def test_stale_lock_file_does_not_block_a_new_writer(self) -> None:
        lock_path = index_lock_path(self.manifest)
        lock_path.parent.mkdir(parents=True)
        lock_path.write_text("stale diagnostic data", encoding="utf-8")

        with acquire_index_lock(self.manifest):
            self.assertTrue(lock_path.exists())

    def test_invalid_timeout_is_rejected(self) -> None:
        for timeout in (-1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    with acquire_index_lock(self.manifest, timeout_seconds=timeout):
                        pass


if __name__ == "__main__":
    unittest.main()
