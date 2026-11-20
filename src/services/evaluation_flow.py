"""Orchestrate queued dataset evaluation and optional MLflow tracking."""

from __future__ import annotations

import asyncpg
from faststream.rabbit import RabbitBroker
from mlflow.tracking import MlflowClient

from src.config import Settings
from src.exceptions import EvaluationTrackingError, InvalidDatasetError
from src.schemas import (
    EvaluationError,
    EvaluationFlowResult,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationTrackingStatus,
)
from src.services.batch_processing import batch_process_evaluate
from src.services.evaluate_unpack import prepare_evaluation_batch, remove_evaluation_batch
from src.services.evaluation_aggregate import build_evaluation_report
from src.services.evaluation_tracking import record_evaluation


async def run_evaluation_flow(
    data: EvaluationRequest,
    *,
    broker: RabbitBroker,
    pool: asyncpg.Pool,
    settings: Settings,
) -> EvaluationFlowResult:
    """Unpack a ZIP, run workers via the evaluate queue, aggregate, and log to MLflow."""
    try:
        prepared = prepare_evaluation_batch(data, settings)
    except InvalidDatasetError as exc:
        return EvaluationFlowResult(
            status_code=400,
            error=EvaluationError(status="invalid_dataset", message=str(exc)),
        )

    try:
        try:
            results = await batch_process_evaluate(prepared.tasks, broker, pool, settings)
        except TimeoutError as exc:
            return EvaluationFlowResult(
                status_code=503,
                error=EvaluationError(status="evaluation_timeout", message=str(exc)),
            )

        first = results[prepared.tasks[0].task_id]
        report = build_evaluation_report(
            dataset_name=prepared.dataset_name,
            dataset_sha256=prepared.dataset_sha256,
            model_ref=first.model_ref,
            model_version=first.model_version,
            tasks=prepared.tasks,
            results=results,
            settings=settings,
        )

        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        try:
            run_id = record_evaluation(client, report)
        except EvaluationTrackingError as exc:
            return EvaluationFlowResult(
                status_code=503,
                response=EvaluationResponse(
                    dataset_name=report.dataset_name,
                    dataset_sha256=report.dataset_sha256,
                    sample_count=len(report.predictions),
                    model=report.model,
                    model_version=report.model_version,
                    metrics=report.metrics,
                    run_id=None,
                    tracking=EvaluationTrackingStatus(
                        status="unavailable",
                        message=str(exc),
                    ),
                ),
            )

        return EvaluationFlowResult(
            status_code=200,
            response=EvaluationResponse(
                dataset_name=report.dataset_name,
                dataset_sha256=report.dataset_sha256,
                sample_count=len(report.predictions),
                model=report.model,
                model_version=report.model_version,
                metrics=report.metrics,
                run_id=run_id,
                tracking=EvaluationTrackingStatus(status="recorded"),
            ),
        )
    finally:
        remove_evaluation_batch(prepared, settings)
