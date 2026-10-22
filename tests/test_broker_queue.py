"""Verify bounded queue arguments and soft/hard settings."""

import pytest
from pydantic import ValidationError

from src.broker import task_queue
from src.config import Settings


def test_task_queue_rejects_publish_when_hard_limit_reached() -> None:
    settings = Settings(task_queue_max_length=32, task_queue_soft_limit=20)
    arguments = task_queue(settings).arguments
    assert arguments is not None
    assert arguments["x-max-length"] == 32
    assert arguments["x-overflow"] == "reject-publish"


def test_settings_reject_soft_limit_above_hard_limit() -> None:
    with pytest.raises(ValidationError):
        Settings(task_queue_max_length=10, task_queue_soft_limit=11)
