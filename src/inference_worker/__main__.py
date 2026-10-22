"""Run the inference worker: uv run python -m src.inference_worker."""

from __future__ import annotations

import asyncio

from src.config import get_settings
from src.inference_worker.consumer import run_inference_worker
from src.logging_config import setup_logging


def main() -> None:
    """Run the inference worker with settings loaded from the environment."""
    settings = get_settings()
    setup_logging(settings.log_level)
    asyncio.run(run_inference_worker(settings))


if __name__ == "__main__":
    main()
