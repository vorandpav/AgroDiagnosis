"""MLflow Tracking probe and optional model load at application startup."""

from __future__ import annotations

import logging
import urllib.error
from typing import NamedTuple

from torch import nn

from src.config import Settings
from src.services import health as health_service
from src.services import inference as inference_service

log = logging.getLogger(__name__)


class MLflowRuntime(NamedTuple):
    """Startup result: Registry reference and optionally loaded model weights."""

    model_ref: str
    model: nn.Module | None
    class_names: list[str]


def setup_mlflow(settings: Settings) -> MLflowRuntime:
    """Probe Tracking and load the Registry model when available.

    Logs Tracking reachability and model load as separate messages. The
    application may start degraded when Tracking is down or the alias is missing.

    Args:
        settings: Application settings with Tracking URI and model identity.

    Returns:
        MLflowRuntime with ``model`` set on success, otherwise ``model=None``.
    """
    model_ref = f"{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    tracking_uri = settings.mlflow_tracking_uri

    try:
        health_service.probe_mlflow_tracking(tracking_uri)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        log.warning("MLflow Tracking Server unreachable at %s: %s", tracking_uri, exc)
        log.warning("Inference model not loaded (%s): tracking unreachable", model_ref)
        return MLflowRuntime(model_ref=model_ref, model=None, class_names=[])

    log.info("MLflow Tracking Server reachable at %s", tracking_uri)
    try:
        model, class_names = inference_service.load_model(settings)
    except RuntimeError as exc:
        log.warning("Inference model not loaded (%s): %s", model_ref, exc)
        return MLflowRuntime(model_ref=model_ref, model=None, class_names=[])

    log.info("Inference model loaded: %s (%d classes)", model_ref, len(class_names))
    return MLflowRuntime(model_ref=model_ref, model=model, class_names=class_names)
