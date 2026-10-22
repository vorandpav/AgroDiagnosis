"""HTTP API for asynchronous inference tasks."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from src.broker import retry_after_seconds
from src.exceptions import QueueFullError
from src.schemas import TaskAccepted, TaskResult
from src.services import task_submission

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks")


@router.post("", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_task(request: Request, file: UploadFile) -> TaskAccepted | JSONResponse:
    """Accept an image into storage and the bounded inference queue."""
    image_bytes = await file.read()
    try:
        return await task_submission.submit_uploaded_task(
            image_bytes,
            file.filename,
            request.app.state.settings,
            request.app.state.broker,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except QueueFullError:
        retry_after = str(retry_after_seconds(request.app.state.settings))
        log.warning("Rejecting task upload: queue full (Retry-After=%s)", retry_after)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Task queue is full"},
            headers={"Retry-After": retry_after},
        )


@router.get("/{task_id}", response_model=TaskResult)
async def get_task(request: Request, task_id: UUID) -> TaskResult:
    """Return a finished task result or 404 while it is still processing."""
    result = await task_submission.fetch_task_result(task_id, request.app.state.db_pool)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task result not found")
    return result
