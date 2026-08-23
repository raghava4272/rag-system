"""Durable source and configuration state for incremental indexing."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1


class IndexLockError(RuntimeError):
    """Raised when another process already owns an index writer lock."""


@dataclass(frozen=True, slots=True)
class IndexConfiguration:
    embedding_model: str
    chunk_size: int
    overlap: int
    tokenizer: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "tokenizer": self.tokenizer,
        }


@dataclass(frozen=True, slots=True)
class SourceManifest:
    fingerprint: str
    raw_document_count: int
    chunk_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, str | int | list[str]]:
        return {
            "fingerprint": self.fingerprint,
            "raw_document_count": self.raw_document_count,
            "chunk_ids": list(self.chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class IndexManifest:
    configuration: IndexConfiguration
    sources: dict[str, SourceManifest]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "configuration": self.configuration.as_dict(),
            "sources": {
                source: state.as_dict()
                for source, state in sorted(self.sources.items())
            },
        }


def fingerprint_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_manifest_path(chroma_path: str | Path, collection: str) -> Path:
    collection_key = sha256(collection.encode("utf-8")).hexdigest()[:16]
    return Path(chroma_path) / ".manifests" / f"{collection_key}.json"


def index_lock_path(manifest_path: str | Path) -> Path:
    """Return the advisory lock file paired with one collection manifest."""

    path = Path(manifest_path)
    return path.with_name(f"{path.name}.lock")


@contextmanager
def acquire_index_lock(
    manifest_path: str | Path,
    *,
    timeout_seconds: float = 0,
) -> Iterator[None]:
    """Hold an exclusive OS lock for one complete incremental index transaction.

    The file is only a stable rendezvous point. Lock ownership is maintained by
    the operating system, so a process exit releases it even if the file remains.
    """

    if not isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("Index lock timeout must be finite and non-negative")

    lock_path = index_lock_path(manifest_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    acquired = False

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise IndexLockError(
                        f"Indexing is already running for manifest {Path(manifest_path)}"
                    ) from error
                time.sleep(min(0.05, remaining))

        diagnostic = {
            "pid": os.getpid(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        lock_file.seek(0)
        lock_file.truncate()
        json.dump(diagnostic, lock_file, sort_keys=True)
        lock_file.write("\n")
        lock_file.flush()
        os.fsync(lock_file.fileno())

        try:
            yield
        finally:
            if acquired:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _invalid(path: Path, detail: str) -> ValueError:
    return ValueError(f"Index manifest {path} is invalid: {detail}")


def load_manifest(path: str | Path) -> IndexManifest | None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _invalid(manifest_path, "expected valid JSON") from error

    if not isinstance(payload, dict) or payload.get("version") != MANIFEST_VERSION:
        raise _invalid(manifest_path, f"expected version {MANIFEST_VERSION}")
    raw_configuration = payload.get("configuration")
    raw_sources = payload.get("sources")
    if not isinstance(raw_configuration, dict) or not isinstance(raw_sources, dict):
        raise _invalid(manifest_path, "missing configuration or sources object")
    try:
        configuration = IndexConfiguration(
            embedding_model=raw_configuration["embedding_model"],
            chunk_size=raw_configuration["chunk_size"],
            overlap=raw_configuration["overlap"],
            tokenizer=raw_configuration["tokenizer"],
        )
    except (KeyError, TypeError) as error:
        raise _invalid(manifest_path, "malformed configuration") from error
    if (
        not isinstance(configuration.embedding_model, str)
        or not isinstance(configuration.chunk_size, int)
        or not isinstance(configuration.overlap, int)
        or not isinstance(configuration.tokenizer, str)
    ):
        raise _invalid(manifest_path, "malformed configuration values")

    sources: dict[str, SourceManifest] = {}
    for source, raw_state in raw_sources.items():
        if not isinstance(source, str) or not isinstance(raw_state, dict):
            raise _invalid(manifest_path, "malformed source entry")
        fingerprint = raw_state.get("fingerprint")
        raw_document_count = raw_state.get("raw_document_count")
        chunk_ids = raw_state.get("chunk_ids")
        if (
            not isinstance(fingerprint, str)
            or not isinstance(raw_document_count, int)
            or raw_document_count < 0
            or not isinstance(chunk_ids, list)
            or not all(isinstance(record_id, str) for record_id in chunk_ids)
        ):
            raise _invalid(manifest_path, f"malformed source state for {source!r}")
        sources[source] = SourceManifest(
            fingerprint=fingerprint,
            raw_document_count=raw_document_count,
            chunk_ids=tuple(chunk_ids),
        )
    return IndexManifest(configuration=configuration, sources=sources)


def write_manifest(path: str | Path, manifest: IndexManifest) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    serialized = json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n"
    try:
        with temporary_path.open("w", encoding="utf-8") as temporary_file:
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
