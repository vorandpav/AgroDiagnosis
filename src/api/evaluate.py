"""Evaluation endpoint for the loaded model on a new labelled test dataset."""

from typing import Annotated

from fastapi import APIRouter, Form, Request, UploadFile
from mlflow.tracking import MlflowClient
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from src.exceptions import EvaluationTrackingError, InvalidDatasetError
from src.schemas import EvaluationError, EvaluationRequest, EvaluationResponse
from src.services.evaluation import evaluate_dataset
from src.services.evaluation_tracking import record_evaluation

router = APIRouter()


def _error(status_code: int, status: str, message: str) -> JSONResponse:
    """Build a stable machine-readable error response."""
    error = EvaluationError(status=status, message=message)
    return JSONResponse(status_code=status_code, content=error.model_dump())


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    responses={
        400: {"model": EvaluationError},
        503: {"model": EvaluationError},
    },
)
async def evaluate(
    request: Request,
    file: UploadFile,
    dataset_name: Annotated[
        str, Form(min_length=1, max_length=100, pattern=r"^[\w .-]+$")
    ] = "uploaded-test",
) -> EvaluationResponse | JSONResponse:
    """Evaluate a ZIP with Plant/Condition folders and log results to MLflow."""
    state = request.app.state
    if state.model is None or state.model_version is None:
        return _error(503, "model_unavailable", "Inference model is unavailable")
    archive = await file.read()
    if not archive:
        return _error(400, "invalid_dataset", "ZIP archive is empty")
    data = EvaluationRequest(dataset_name=dataset_name, archive=archive)
    try:
        report = await run_in_threadpool(
            evaluate_dataset,
            data,
            state.model,
            state.class_names,
            state.model_ref,
            state.model_version,
        )
    except InvalidDatasetError as exc:
        return _error(400, "invalid_dataset", str(exc))
    client = MlflowClient(tracking_uri=state.settings.mlflow_tracking_uri)
    try:
        run_id = await run_in_threadpool(record_evaluation, client, report)
    except EvaluationTrackingError as exc:
        return _error(503, "tracking_unavailable", str(exc))
    return EvaluationResponse(
        dataset_name=report.dataset_name,
        dataset_sha256=report.dataset_sha256,
        sample_count=len(report.predictions),
        model=report.model,
        model_version=report.model_version,
        metrics=report.metrics,
        run_id=run_id,
    )
