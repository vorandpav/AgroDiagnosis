"""Tests for evaluation tracking and model version pinning."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking import MlflowClient
from torch import nn

from src.config import Settings
from src.exceptions import EvaluationTrackingError
from src.schemas import EvaluationMetrics, EvaluationPrediction, EvaluationReport
from src.services.evaluation_tracking import record_evaluation
from src.services.inference import load_model


def make_report() -> EvaluationReport:
    return EvaluationReport(
        dataset_name="fresh-test",
        dataset_sha256="a" * 64,
        model="plants@champion",
        model_version="7",
        metrics=EvaluationMetrics(accuracy=1, precision_macro=1, recall_macro=1, f1_macro=1),
        labels=["Apple/healthy"],
        confusion_matrix=[[1]],
        predictions=[
            EvaluationPrediction(
                path="Apple/healthy/a.png",
                sha256="b" * 64,
                expected="Apple/healthy",
                predicted="Apple/healthy",
                confidence=0.9,
            )
        ],
    )


def tracking_client() -> MagicMock:
    client = MagicMock(spec=MlflowClient)
    client.get_experiment_by_name.return_value.experiment_id = "experiment"
    client.create_run.return_value.info.run_id = "run"
    return client


def test_record_evaluation_logs_dataset_metrics_and_predictions() -> None:
    client = tracking_client()
    report = make_report()
    assert record_evaluation(client, report) == "run"
    dataset = client.log_inputs.call_args.kwargs["datasets"][0]
    assert dataset.dataset.digest == "a" * 32
    assert dataset.tags[0].value == "test"
    assert client.log_metric.call_count == 4
    client.log_dict.assert_called_once_with(
        "run", report.model_dump(mode="json"), "evaluation.json"
    )
    client.set_terminated.assert_called_once_with("run", status="FINISHED")
    assert client.create_run.call_args.kwargs["tags"]["model_version"] == "7"


def test_record_evaluation_creates_experiment_when_missing() -> None:
    client = tracking_client()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "new-experiment"
    record_evaluation(client, make_report())
    client.create_experiment.assert_called_once_with("AgroDiagnosisEvaluation")
    assert client.create_run.call_args.args == ("new-experiment",)


def test_record_evaluation_marks_failed_run_when_logging_fails() -> None:
    client = tracking_client()
    client.log_inputs.side_effect = MlflowException("private connection details")
    with pytest.raises(EvaluationTrackingError, match="tracking is unavailable") as error:
        record_evaluation(client, make_report())
    assert isinstance(error.value.__cause__, MlflowException)
    client.set_terminated.assert_called_once_with("run", status="FAILED")


def test_record_evaluation_reports_failure_before_run_created() -> None:
    client = tracking_client()
    client.get_experiment_by_name.side_effect = MlflowException("private connection details")
    with pytest.raises(EvaluationTrackingError, match="tracking is unavailable"):
        record_evaluation(client, make_report())
    client.log_inputs.assert_not_called()


def test_load_model_pins_resolved_version_instead_of_mutable_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = tracking_client()
    client.get_model_version_by_alias.return_value.version = "7"
    model = nn.Identity()
    loader = MagicMock(return_value=model)
    monkeypatch.setattr("src.services.inference.probe_mlflow_tracking", lambda uri: None)
    monkeypatch.setattr("src.services.inference.MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr("src.services.inference.mlflow.set_tracking_uri", lambda uri: None)
    monkeypatch.setattr("src.services.inference.mlflow.pytorch.load_model", loader)
    names = tmp_path / "classes.json"
    names.write_text(json.dumps(["Apple/healthy"]), encoding="utf-8")
    monkeypatch.setattr("src.services.inference._CLASS_NAMES_PATH", names)
    result = load_model(Settings(mlflow_model_name="plants"))
    loader.assert_called_once_with("models:/plants/7")
    assert result == (model, ["Apple/healthy"], "7")
    assert not model.training


def test_record_evaluation_handles_concurrent_experiment_creation() -> None:
    client = tracking_client()
    experiment = MagicMock(experiment_id="concurrent-experiment")
    client.get_experiment_by_name.side_effect = [None, experiment]
    client.create_experiment.side_effect = MlflowException(
        "Already created", error_code=RESOURCE_ALREADY_EXISTS
    )
    assert record_evaluation(client, make_report()) == "run"
    assert client.create_run.call_args.args == ("concurrent-experiment",)


def test_record_evaluation_preserves_original_error_when_failure_cleanup_fails() -> None:
    client = tracking_client()
    original = MlflowException("Cannot log input")
    client.log_inputs.side_effect = original
    client.set_terminated.side_effect = MlflowException("Cannot close run")
    with pytest.raises(EvaluationTrackingError) as error:
        record_evaluation(client, make_report())
    assert error.value.__cause__ is original
