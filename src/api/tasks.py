"""HTTP API for asynchronous inference tasks."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from src.api.deps import BrokerDep, PoolDep, SettingsDep
from src.exceptions import QueueFullError
from src.messaging.broker import retry_after_seconds
from src.schemas import TaskAccepted, TaskResult
from src.services import tasks as tasks_service
from src.telemetry import record_span_exception

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks")


@router.post("", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    file: UploadFile,
    settings: SettingsDep,
    broker: BrokerDep,
) -> TaskAccepted | JSONResponse:
    """Accept an image into storage and the bounded inference queue."""
    image_bytes = await file.read()
    try:
        return await tasks_service.submit_uploaded_task(
            image_bytes,
            file.filename,
            settings,
            broker,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except QueueFullError as exc:
        record_span_exception(exc)
        retry_after = str(retry_after_seconds(settings))
        log.warning("Rejecting task upload: queue full (Retry-After=%s)", retry_after)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Task queue is full"},
            headers={"Retry-After": retry_after},
        )


@router.get("/{task_id}", response_model=TaskResult)
async def get_task(task_id: UUID, pool: PoolDep) -> TaskResult:
    """Return a finished task result or 404 while it is still processing."""
    result = await tasks_service.get_task_result(task_id, pool)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task result not found")
    return result
