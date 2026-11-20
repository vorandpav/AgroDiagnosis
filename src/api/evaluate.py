"""Evaluation endpoint: queue-backed batch inference on uploaded ZIP datasets."""

from typing import Annotated

from fastapi import APIRouter, Form, UploadFile
from starlette.responses import JSONResponse

from src.api.deps import BrokerDep, PoolDep, SettingsDep
from src.schemas import (
    EvaluationError,
    EvaluationFlowResult,
    EvaluationRequest,
    EvaluationResponse,
)
from src.services.evaluation_flow import run_evaluation_flow

router = APIRouter()


def _json_result(result: EvaluationFlowResult) -> JSONResponse:
    """Serialize a flow result to JSON with the chosen HTTP status."""
    if result.response is not None:
        return JSONResponse(
            status_code=result.status_code,
            content=result.response.model_dump(),
        )
    if result.error is not None:
        return JSONResponse(
            status_code=result.status_code,
            content=result.error.model_dump(),
        )
    raise RuntimeError("EvaluationFlowResult must include response or error")


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    responses={
        400: {"model": EvaluationError},
        503: {"model": EvaluationResponse},
    },
)
async def evaluate(
    file: UploadFile,
    broker: BrokerDep,
    pool: PoolDep,
    settings: SettingsDep,
    dataset_name: Annotated[
        str, Form(min_length=1, max_length=100, pattern=r"^[\w .-]+$")
    ] = "uploaded-test",
) -> EvaluationResponse | JSONResponse:
    """Evaluate a ZIP with Plant/Condition folders via workers and log to MLflow."""
    archive = await file.read()
    if not archive:
        return JSONResponse(
            status_code=400,
            content=EvaluationError(
                status="invalid_dataset",
                message="ZIP archive is empty",
            ).model_dump(),
        )
    data = EvaluationRequest(dataset_name=dataset_name, archive=archive)
    result = await run_evaluation_flow(
        data,
        broker=broker,
        pool=pool,
        settings=settings,
    )
    return _json_result(result)
