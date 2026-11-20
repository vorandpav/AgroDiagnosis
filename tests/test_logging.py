"""Structured JSON logs and task correlation."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.config import Settings
from src.db.database import create_database
from src.logging_config import HealthAccessFilter, setup_logging, task_id_context

_REQUIRED_KEYS = {
    "timestamp",
    "level",
    "service",
    "logger",
    "message",
    "request_id",
    "task_id",
    "trace_id",
    "span_id",
}


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Put the process loggers back after a test reconfigures them."""
    root = logging.getLogger()
    root_level = root.level
    root_handlers = root.handlers[:]
    uvicorn_names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    saved = {
        name: (
            logging.getLogger(name).level,
            logging.getLogger(name).handlers[:],
            logging.getLogger(name).propagate,
        )
        for name in uvicorn_names
    }
    yield
    root.setLevel(root_level)
    root.handlers = root_handlers
    for name, (level, handlers, propagate) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers = handlers
        logger.propagate = propagate


def _json_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_setup_logging_emits_required_json_fields_and_respects_level(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    logger = logging.getLogger("tests.logging.json")
    setup_logging("DEBUG")
    logger.debug("debug-visible")
    setup_logging("INFO")
    logger.debug("debug-hidden")
    logger.info("info-visible")

    records = _json_lines(capsys.readouterr().out)
    assert [record["message"] for record in records] == ["debug-visible", "info-visible"]
    assert [record["level"] for record in records] == ["DEBUG", "INFO"]
    for record in records:
        assert set(record) == _REQUIRED_KEYS
        assert record["service"] == "agrodiagnosis"
        assert record["logger"] == "tests.logging.json"
        assert record["request_id"] == "-"
        assert record["task_id"] == "-"
        assert record["trace_id"] == "-"
        assert record["span_id"] == "-"
        datetime.fromisoformat(str(record["timestamp"]))


def test_task_id_context_appears_without_message_interpolation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging("INFO")
    task_id = "11111111-1111-1111-1111-111111111111"
    with task_id_context(task_id):
        logging.getLogger("tests.logging.task").info("Accepted task")

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["task_id"] == task_id
    assert payload["message"] == "Accepted task"
    assert task_id not in payload["message"]


async def test_pool_log_omits_password_and_create_database_logs_host_only(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "postgresql://postgres:postgres@localhost:5433/agrodiagnosis"
    host = "localhost:5433/agrodiagnosis"
    setup_logging("INFO")
    logging.getLogger("src.db.database").info(
        "PostgreSQL connection pool established: %s",
        host,
    )
    emitted = capsys.readouterr().out
    assert "postgres:postgres" not in emitted
    assert url not in emitted
    assert host in emitted

    info = Mock()
    monkeypatch.setattr("src.db.database.asyncpg.create_pool", AsyncMock(return_value=Mock()))
    monkeypatch.setattr("src.db.database.log.info", info)
    await create_database(Settings(database_url=url))
    rendered = " ".join(str(part) for part in info.call_args.args)
    assert url not in rendered
    assert "postgres:postgres" not in rendered
    assert info.call_args.args[1] == host


def _access_record(path: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=level,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:9", "GET", path, "1.1", 200),
        exc_info=None,
    )


def test_health_access_filter_drops_probe_paths_and_keeps_other_paths() -> None:
    access_filter = HealthAccessFilter()
    assert access_filter.filter(_access_record("/healthz")) is False
    assert access_filter.filter(_access_record("/api/v1/health")) is False
    assert access_filter.filter(_access_record("/healthz?probe=1")) is False
    assert access_filter.filter(_access_record("/api/v1/tasks")) is True
    assert access_filter.filter(_access_record("/healthz", logging.WARNING)) is True
