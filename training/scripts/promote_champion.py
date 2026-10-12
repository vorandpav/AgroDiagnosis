"""Promote the best finished training run to Model Registry @champion.

Picks the run with the highest logged ``f1_macro`` and registers its ``model_uri`` tag.
"""

from __future__ import annotations

import logging

# Must import before mlflow: sets env (agent hint / uv scan / progress bars).
from src.mlflow_quiet import configure_mlflow_logging

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from src.register import register_champion
from src.settings import get_settings, resolve_path

configure_mlflow_logging()

log = logging.getLogger(__name__)


def main() -> None:
    """Register the best val f1_macro run as @champion."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    conf = OmegaConf.load(resolve_path(settings.conf_dir) / "config.yaml")
    experiment_name = str(conf.mlflow.experiment_name)
    registered_model_name = str(conf.mlflow.registered_model_name)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {experiment_name!r} not found")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.f1_macro DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No FINISHED runs in experiment {experiment_name!r}")

    run = runs[0]
    f1_macro = run.data.metrics.get("f1_macro")
    model_uri = run.data.tags.get("model_uri")
    if not model_uri and run.outputs and run.outputs.model_outputs:
        model_uri = f"models:/{run.outputs.model_outputs[0].model_id}"
    if not model_uri:
        raise RuntimeError(f"Run {run.info.run_id} has no logged model to promote")

    version = register_champion(model_uri, registered_model_name)
    log.info(
        "Champion f1_macro=%s run=%s -> %s v%s @champion",
        f1_macro,
        run.info.run_id,
        registered_model_name,
        version,
    )


if __name__ == "__main__":
    main()
