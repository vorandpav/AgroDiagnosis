"""Application centralized logging configuration module.

Provides a unified log format for the service and Uvicorn server,
as well as end-to-end HTTP request tracing using request_id.
"""

import contextvars
import logging
import sys

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(request_id)-8s | %(name)s | %(message)s"

_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


class RequestIdFilter(logging.Filter):
    """Filter for automatically adding request_id to the log record object."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Injects request_id from the current execution context."""
        record.request_id = REQUEST_ID.get() or "-"
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configures the root application logger and intercepts Uvicorn logs.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(RequestIdFilter())

    # Configure the root logger
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Intercept and reformat Uvicorn logs
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
