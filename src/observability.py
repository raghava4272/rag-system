"""Small, dependency-free observability primitives for the HTTP service."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import UUID, uuid4


LOG_LEVEL_ENV = "RAG_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_USER_ID: ContextVar[str | None] = ContextVar("user_id", default=None)
_SESSION_ID: ContextVar[str | None] = ContextVar("session_id", default=None)

_CONTEXT_FIELDS = {
    "request_id": _REQUEST_ID,
    "user_id": _USER_ID,
    "session_id": _SESSION_ID,
}
_EVENT_FIELDS = (
    "method",
    "path",
    "status_code",
    "duration_ms",
    "exception_type",
)


def parse_log_level(value: str | None) -> int:
    """Parse a deliberately small set of application log levels."""

    normalized = (value or DEFAULT_LOG_LEVEL).strip().upper()
    try:
        return _LOG_LEVELS[normalized]
    except KeyError as error:
        choices = ", ".join(_LOG_LEVELS)
        raise ValueError(
            f"{LOG_LEVEL_ENV} must be one of: {choices}"
        ) from error


def normalize_request_id(value: str | None) -> str:
    """Accept a UUID correlation ID or generate a safe UUIDv4 replacement."""

    if value:
        try:
            return str(UUID(value))
        except (ValueError, AttributeError):
            pass
    return str(uuid4())


@contextmanager
def bind_log_context(
    *,
    request_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[None]:
    """Bind correlation fields for the current async/thread context."""

    supplied = {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
    }
    tokens = []
    for field, value in supplied.items():
        if value is not None:
            variable = _CONTEXT_FIELDS[field]
            tokens.append((variable, variable.set(value)))
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


class StructuredJsonFormatter(logging.Formatter):
    """Serialize an allowlisted application event as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        event: dict[str, object] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        for field, variable in _CONTEXT_FIELDS.items():
            value = getattr(record, field, None) or variable.get()
            if value is not None:
                event[field] = value

        for field in _EVENT_FIELDS:
            value = getattr(record, field, None)
            if value is None:
                continue
            if field == "duration_ms":
                value = round(float(value), 3)
            event[field] = value

        return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def configure_application_logging(
    level: str | None = None,
    *,
    logger: logging.Logger | None = None,
) -> logging.Logger:
    """Configure the application logger once while allowing level updates."""

    configured_logger = logger or logging.getLogger("rag")
    configured_logger.setLevel(parse_log_level(level or os.getenv(LOG_LEVEL_ENV)))
    configured_logger.propagate = False

    for handler in configured_logger.handlers:
        if getattr(handler, "_rag_structured_handler", False):
            return configured_logger

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter())
    handler._rag_structured_handler = True  # type: ignore[attr-defined]
    configured_logger.addHandler(handler)
    return configured_logger
