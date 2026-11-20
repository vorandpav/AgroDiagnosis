"""Main FastAPI application construction and lifespan configuration."""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.metrics import NoOpMeterProvider
from starlette.responses import Response

from src import config, logging_config, telemetry
from src.api import evaluate, health, hello, tasks
from src.db.database import create_database
from src.db.task_results import initialize_results
from src.messaging.broker import (
    create_broker,
    dead_letter_queue,
    evaluate_dead_letter_queue,
    evaluate_task_queue,
    task_queue,
)

log = logging.getLogger(__name__)


def _http_route(request: Request) -> str:
    """Return the matched route template."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return "unmatched"


def create_app() -> FastAPI:
    """Assemble the FastAPI app"""
    settings = config.get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage pool, broker, and cleanup."""
        logging_config.setup_logging(settings.log_level)
        telemetry.setup_telemetry(settings.inference_min_seconds)

        app.state.settings = settings

        database = await create_database(settings)
        await database.declare_table(initialize_results)
        app.state.db_pool = database.pool

        broker = create_broker(settings)
        await broker.connect()
        await broker.declare_queue(dead_letter_queue(settings))
        await broker.declare_queue(task_queue(settings))
        await broker.declare_queue(evaluate_dead_letter_queue(settings))
        await broker.declare_queue(evaluate_task_queue(settings))
        app.state.broker = broker

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
            await database.close()
            log.info("Application shut down successfully.")

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
        response: Response | None = None

        try:
            response = await call_next(request)
            if request.url.path not in logging_config.QUIET_INFO_PATHS:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                log.info(
                    "%s %s -> %d (%.1f ms)",
                    request.method,
                    request.url.path,
                    response.status_code,
                    elapsed_ms,
                )
        finally:
            if response is not None:
                telemetry.record_api_request(
                    _http_route(request),
                    response.status_code,
                    time.perf_counter() - start_time,
                )
            logging_config.REQUEST_ID.reset(token)

        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(hello.router)
    app.include_router(health.router)
    app.include_router(tasks.router)
    app.include_router(evaluate.router)
    FastAPIInstrumentor.instrument_app(app, meter_provider=NoOpMeterProvider())
    return app


app = create_app()
