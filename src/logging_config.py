"""JSON logs on stdout, with request and task correlation."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
TASK_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)

QUIET_INFO_PATHS: frozenset[str] = frozenset({"/healthz", "/api/v1/health"})

_EXTRA_FIELDS = ("alias", "model_version", "attempt")
_ACCESS_LOGGER = "uvicorn.access"
_MISSING = "-"


@contextmanager
def task_id_context(task_id: str) -> Iterator[None]:
    """Bind task_id onto log records emitted in this block."""
    token = TASK_ID.set(task_id)
    try:
        yield
    finally:
        TASK_ID.reset(token)


def _service_name() -> str:
    """Return the service name from OTEL_SERVICE_NAME."""
    return os.environ.get("OTEL_SERVICE_NAME") or "agrodiagnosis"


def _trace_and_span_ids() -> tuple[str, str]:
    """Return the current span ids, or dashes when none is active."""
    try:
        from opentelemetry import trace
    except ImportError:
        return _MISSING, _MISSING
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return _MISSING, _MISSING
    return f"{span_context.trace_id:032x}", f"{span_context.span_id:016x}"


def _context_value(context: contextvars.ContextVar[str | None]) -> str:
    """Return a context value, or a dash when it is unset."""
    return context.get() or _MISSING


def _extra_value(record: logging.LogRecord, name: str) -> str | int | float | None:
    """Return a whitelisted extra when the record carries a scalar."""
    if name not in record.__dict__:
        return None
    value = record.__dict__[name]
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    return value


def _exception_text(record: logging.LogRecord) -> str:
    """Format exc_info as a traceback string."""
    exc_info = record.exc_info
    if exc_info is None or exc_info[0] is None:
        return ""
    return "".join(traceback.format_exception(*exc_info)).rstrip()


class JsonLogFormatter(logging.Formatter):
    """Format one log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the record, context, and whitelisted extras."""
        trace_id, span_id = _trace_and_span_ids()
        payload: dict[str, str | int | float] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "service": _service_name(),
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _context_value(REQUEST_ID),
            "task_id": _context_value(TASK_ID),
            "trace_id": trace_id,
            "span_id": span_id,
        }
        for name in _EXTRA_FIELDS:
            value = _extra_value(record, name)
            if value is not None:
                payload[name] = value
        if record.exc_info:
            payload["exception"] = _exception_text(record)
        return json.dumps(payload, ensure_ascii=False)


def _access_path(record: logging.LogRecord) -> str:
    """Return the URL path from a uvicorn access record."""
    args = record.args
    if not isinstance(args, tuple) or len(args) < 3:
        return ""
    path = args[2]
    if not isinstance(path, str):
        return ""
    return path.split("?", 1)[0]


class HealthAccessFilter(logging.Filter):
    """Drop INFO access logs for liveness and health probes."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep the record unless it is an INFO probe access line."""
        if record.name != _ACCESS_LOGGER or record.levelno != logging.INFO:
            return True
        return _access_path(record) not in QUIET_INFO_PATHS


def setup_logging(level: str = "INFO") -> None:
    """Attach one JSON stdout handler to the root and uvicorn loggers.

    Args:
        level: Root log level name.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    handler.addFilter(HealthAccessFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
