"""Run a standalone FastStream inference worker: uv run python -m src.inference_worker."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from functools import partial

from faststream import FastStream
from faststream.middlewares import AckPolicy
from faststream.rabbit import RabbitBroker

from src.broker import create_broker, dead_letter_queue, task_queue
from src.config import Settings, get_settings
from src.db import close_db_pool, create_db_pool
from src.logging_config import setup_logging
from src.schemas import InferenceTask, TaskResult
from src.services.inference import load_model
from src.services.tasks import initialize_results, process_task, save_result

log = logging.getLogger(__name__)


def register_handler(
    broker: RabbitBroker,
    settings: Settings,
    execute: Callable[[InferenceTask], Awaitable[TaskResult]],
    save: Callable[[TaskResult], Awaitable[None]],
) -> None:
    """Register a handler with explicit inference and persistence dependencies."""

    @broker.subscriber(task_queue(settings), ack_policy=AckPolicy.REJECT_ON_ERROR)
    async def handle(task: InferenceTask) -> None:
        """Save the result before acknowledging successful processing."""
        result = await execute(task)
        await save(result)
        log.info("Task %s saved (%.3fs)", task.task_id, result.processing_seconds)


async def run_inference_worker(settings: Settings) -> None:
    """Load the model once and release resources after the broker stops."""
    async with AsyncExitStack() as resources:
        executor = resources.enter_context(ThreadPoolExecutor(max_workers=1))
        loop = asyncio.get_running_loop()
        model, class_names, model_version = await loop.run_in_executor(
            executor, load_model, settings
        )
        log.info(
            "Loaded %s@%s version %s",
            settings.mlflow_model_name,
            settings.mlflow_model_alias,
            model_version,
        )
        pool = await create_db_pool(settings)
        resources.push_async_callback(close_db_pool, pool)
        await initialize_results(pool)
        broker = create_broker(settings)

        async def execute(task: InferenceTask) -> TaskResult:
            """Run file reads, inference, and the delay in a single executor thread."""
            return await loop.run_in_executor(
                executor,
                partial(process_task, task, settings, model, class_names, model_version),
            )

        register_handler(broker, settings, execute, partial(save_result, pool=pool))
        app = FastStream(broker)

        @app.on_startup
        async def prepare_queue() -> None:
            """Declare the dead-letter queue before consuming tasks."""
            await broker.connect()
            await broker.declare_queue(dead_letter_queue(settings))

        await app.run()


def main() -> None:
    """Run the inference worker with settings loaded from the environment."""
    settings = get_settings()
    setup_logging(settings.log_level)
    asyncio.run(run_inference_worker(settings))


if __name__ == "__main__":
    main()
