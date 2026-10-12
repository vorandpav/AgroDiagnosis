"""Training settings from training/.env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

TRAINING_ROOT = Path(__file__).resolve().parents[1]


class TrainingSettings(BaseSettings):
    """Paths and MLflow URI for local runs."""

    model_config = SettingsConfigDict(
        env_file=TRAINING_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    conf_dir: str = Field(default="conf")
    raw_data_dir: str = Field(default="data/plant-diseases")
    data_dir: str = Field(default="data/plant-diseases-processed")


def get_settings() -> TrainingSettings:
    """Load settings from env / .env."""
    return TrainingSettings()


def resolve_path(raw: str) -> Path:
    """Resolve a path from .env relative to training/."""
    path = Path(raw.strip())
    if not path.is_absolute():
        path = TRAINING_ROOT / path
    return path.resolve()
