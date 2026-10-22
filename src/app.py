"""Main FastAPI application construction and lifespan configuration.

Execution:
    `uv run uvicorn src.app:app --reload`
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import Response

from src import config, db, logging_config, mlflow
from src.api import evaluate, health, hello, tasks
from src.broker import create_broker, dead_letter_queue, task_queue
from src.services.tasks import initialize_results

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage pool, broker, optional model load, and cleanup."""
    settings = config.get_settings()
    logging_config.setup_logging(settings.log_level)

    app.state.settings = settings
    app.state.db_pool = await db.create_db_pool(settings)
    await initialize_results(app.state.db_pool)

    broker = create_broker(settings)
    await broker.connect()
    await broker.declare_queue(dead_letter_queue(settings))
    await broker.declare_queue(task_queue(settings))
    app.state.broker = broker

    runtime = mlflow.setup_mlflow(settings)
    app.state.model = runtime.model
    app.state.class_names = runtime.class_names
    app.state.model_ref = runtime.model_ref
    app.state.model_version = runtime.model_version

    log.info(
        "Application %s v%s started (debug=%s)",
        settings.app_name,
        settings.version,
        settings.debug,
    )
    try:
        yield
    finally:
        await broker.close()
        await db.close_db_pool(app.state.db_pool)
        log.info("Application shut down successfully.")


def create_app() -> FastAPI:
    """Assembles and configures the FastAPI application instance.

    Returns:
        FastAPI: Fully configured web application instance.
    """
    settings = config.get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Injects request_id into context and records request processing time."""
        request_id = uuid.uuid4().hex[:8]
        token = logging_config.REQUEST_ID.set(request_id)
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            log.info(
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        finally:
            logging_config.REQUEST_ID.reset(token)

        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(hello.router)
    app.include_router(health.router)
    app.include_router(tasks.router)
    app.include_router(evaluate.router)
    return app


app = create_app()
