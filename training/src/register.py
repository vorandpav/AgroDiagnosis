"""Register an MLflow run model and assign the champion alias."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient


def register_champion(model_uri: str, registered_model_name: str) -> str:
    """Register a logged model URI and set alias ``champion``.

    Args:
        model_uri: URI of the logged model (``runs:/...`` or ``models:/m-...``).
        registered_model_name: Model Registry name.

    Returns:
        Registered model version string.
    """
    try:
        result = mlflow.register_model(model_uri, registered_model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to register model {registered_model_name!r} from {model_uri}"
        ) from exc

    client = MlflowClient()
    try:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=result.version,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to set alias @champion on {registered_model_name} v{result.version}"
        ) from exc

    return str(result.version)
