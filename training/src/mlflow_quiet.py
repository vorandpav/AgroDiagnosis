"""Reduce MLflow console noise during local training runs."""

from __future__ import annotations

import logging
import os
import warnings


def apply_mlflow_env_quiet() -> None:
    """Set env vars that must exist before heavy MLflow operations.

    Call as early as possible (before ``import mlflow`` when feasible).
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # Do not scan/export the whole uv project on every log_model.
    os.environ.setdefault("MLFLOW_UV_AUTO_DETECT", "false")
    os.environ.setdefault("MLFLOW_LOG_UV_FILES", "false")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")


def configure_mlflow_logging() -> None:
    """Drop INFO spam and known harmless UserWarnings from MLflow."""
    apply_mlflow_env_quiet()
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    logging.getLogger("mlflow.utils.uv_utils").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message=r"The specified dataset source can be interpreted in multiple ways",
        category=UserWarning,
        module=r"mlflow\.data\.dataset_source_registry",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Hint: Inferred schema contains integer column",
        category=UserWarning,
        module=r"mlflow\.types\.utils",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Saving pytorch model by Pickle",
        category=UserWarning,
    )


# Apply env on import so ``train.py`` can keep normal import order.
apply_mlflow_env_quiet()
