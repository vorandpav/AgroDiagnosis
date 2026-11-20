"""MLflow Tracking probe and Registry model loading for the inference worker."""

from __future__ import annotations

import json
import logging
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import mlflow
import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from torch import nn

from src.config import Settings

log = logging.getLogger(__name__)

_MLFLOW_PROBE_TIMEOUT_SEC = 3.0


class LoadedModel(NamedTuple):
    """Weights and the Registry identity resolved at load time."""

    model: nn.Module
    class_names: list[str]
    model_ref: str
    model_version: str


def probe_mlflow_tracking(tracking_uri: str) -> None:
    """Probe MLflow Tracking Server liveness endpoint.

    Args:
        tracking_uri: Base Tracking Server URI (e.g. ``http://localhost:5000``).

    Raises:
        OSError: Network failure while contacting the server.
        TimeoutError: Probe exceeded the timeout.
        urllib.error.URLError: HTTP client failure.
    """
    url = f"{tracking_uri.rstrip('/')}/health"
    with urllib.request.urlopen(url, timeout=_MLFLOW_PROBE_TIMEOUT_SEC):
        pass


def init_mlflow_tracking(tracking_uri: str) -> None:
    """Verify Tracking reachability and set the active MLflow client URI."""
    try:
        probe_mlflow_tracking(tracking_uri)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(f"MLflow Tracking Server unreachable at {tracking_uri}") from exc
    mlflow.set_tracking_uri(tracking_uri)
    log.info("MLflow Tracking Server reachable at %s", tracking_uri)


def _class_names_from_run(client: MlflowClient, version: ModelVersion) -> list[str]:
    """Load ordered class labels logged with the registered model run."""
    run_id = version.run_id
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Registered model version has no training run_id for class_names")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_path = client.download_artifacts(run_id, "class_names.json", dst_path=tmp)
            raw = json.loads(Path(local_path).read_text(encoding="utf-8"))
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError("Failed to load class_names.json from the model training run") from exc
    if isinstance(raw, dict):
        names = raw.get("class_names")
        if isinstance(names, list) and names and all(isinstance(name, str) for name in names):
            return names
    raise RuntimeError("class_names.json artifact is missing a non-empty class_names list")


def load_model(settings: Settings) -> LoadedModel:
    """Load the Registry model once and class names for label mapping.

    Resolves the alias to a concrete Registry version at load time so the worker
    process pins that version for all inference until restart.

    Args:
        settings: Application settings with MLflow URI and model identity.

    Returns:
        Loaded weights, class names, the versioned Registry URI, and the version.
    """
    tracking_uri = settings.mlflow_tracking_uri
    init_mlflow_tracking(tracking_uri)
    try:
        version = MlflowClient(tracking_uri=tracking_uri).get_model_version_by_alias(
            settings.mlflow_model_name, settings.mlflow_model_alias
        )
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError("Failed to resolve inference model version") from exc
    uri = f"models:/{settings.mlflow_model_name}/{version.version}"
    try:
        model = mlflow.pytorch.load_model(uri)
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError(f"Failed to load model from {uri}") from exc
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to load model from {uri}") from exc

    model.eval()
    client = MlflowClient(tracking_uri=tracking_uri)
    class_names = _class_names_from_run(client, version)
    resolved_version = str(version.version)
    return LoadedModel(
        model=model,
        class_names=class_names,
        model_ref=uri,
        model_version=resolved_version,
    )
