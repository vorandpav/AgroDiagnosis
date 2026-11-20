"""Runs the inference worker."""

from __future__ import annotations

import asyncio

from src import config, logging_config
from src.inference_worker.consumer import run_inference_worker


def main() -> None:
    """Run the inference worker with settings loaded from the environment."""
    settings = config.get_settings()
    logging_config.setup_logging(settings.log_level)
    asyncio.run(run_inference_worker(settings))


if __name__ == "__main__":
    main()
