"""Record evaluation runs using an explicitly provided MLflow client."""

from __future__ import annotations

import json
import logging

from mlflow.entities import Dataset, DatasetInput, InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from requests.exceptions import RequestException

from src.exceptions import EvaluationTrackingError
from src.schemas import EvaluationReport

log = logging.getLogger(__name__)
_EXPERIMENT_NAME = "AgroDiagnosisEvaluation"
_TRACKING_ERRORS = (MlflowException, RequestException, OSError)


def _create_run(client: MlflowClient, report: EvaluationReport) -> str:
    """Create a run in the evaluation experiment."""
    experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
    if experiment is None:
        try:
            experiment_id = client.create_experiment(_EXPERIMENT_NAME)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
            if experiment is None:
                raise EvaluationTrackingError("Evaluation tracking is unavailable") from exc
            experiment_id = experiment.experiment_id
    else:
        experiment_id = experiment.experiment_id
    run = client.create_run(
        experiment_id,
        run_name=f"evaluate-{report.dataset_name}",
        tags={
            "task": "evaluation",
            "model": report.model,
            "model_version": report.model_version,
            "dataset_sha256": report.dataset_sha256,
        },
    )
    return run.info.run_id


def _write_report(client: MlflowClient, run_id: str, report: EvaluationReport) -> None:
    """Log the dataset, metrics, confusion matrix, and predictions."""
    dataset = Dataset(
        name=report.dataset_name,
        digest=report.dataset_sha256[:32],
        source_type="code",
        source=json.dumps({"tags": {"mlflow.source.name": "POST /evaluate"}}),
        profile=json.dumps({"num_rows": len(report.predictions)}),
    )
    client.log_inputs(
        run_id,
        datasets=[DatasetInput(dataset, tags=[InputTag("mlflow.data.context", "test")])],
    )
    for key, value in report.metrics.model_dump().items():
        client.log_metric(run_id, key, value)
    client.log_param(run_id, "sample_count", len(report.predictions))
    client.log_dict(run_id, report.model_dump(mode="json"), "evaluation.json")
    client.set_terminated(run_id, status="FINISHED")


def record_evaluation(client: MlflowClient, report: EvaluationReport) -> str:
    """Record evaluation results and return the run ID or a stable tracking error."""
    try:
        run_id = _create_run(client, report)
    except _TRACKING_ERRORS as exc:
        log.exception("Cannot create evaluation run")
        raise EvaluationTrackingError("Evaluation tracking is unavailable") from exc
    try:
        _write_report(client, run_id, report)
    except _TRACKING_ERRORS as exc:
        log.exception("Cannot save evaluation run %s", run_id)
        try:
            client.set_terminated(run_id, status="FAILED")
        except _TRACKING_ERRORS:
            log.exception("Cannot mark evaluation run %s as failed", run_id)
        raise EvaluationTrackingError("Evaluation tracking is unavailable") from exc
    return run_id
