"""Application configuration module.

Contains project settings loaded from environment variables and .env configuration files.
"""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings gathered from env / .env.

    Attributes:
        app_name: The name of the web application.
        version: Current semantic version (SemVer).
        log_level: Logging level (DEBUG, INFO, WARNING, etc.).
        debug: Debug mode flag.
        environment: Deployment environment (development, production).
        database_url: PostgreSQL connection string in asyncpg format.
        mlflow_tracking_uri: MLflow Tracking Server URI.
        mlflow_model_name: Registered model name in Model Registry.
        mlflow_model_alias: Alias resolved at startup (e.g. champion).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "agrodiagnosis"
    version: str = "0.1.0"
    log_level: str = "INFO"
    debug: bool = False
    environment: str = "development"
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5433/agrodiagnosis",
        description="PostgreSQL connection string (asyncpg format).",
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow Tracking Server URI.",
    )
    mlflow_model_name: str = Field(
        default="agrodiagnosis-plant-disease",
        description="Registered model name in MLflow Model Registry.",
    )
    mlflow_model_alias: str = Field(
        default="champion",
        description="Model Registry alias loaded at application startup.",
    )
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    task_queue: str = "agrodiagnosis.inference.v1"
    task_queue_max_length: int = Field(
        default=100,
        gt=0,
        description="Hard queue depth; overflow uses reject-publish.",
    )
    task_queue_soft_limit: int = Field(
        default=60,
        gt=0,
        description=(
            "Batch publisher waits at this depth so interactive HTTP tasks "
            "keep headroom up to task_queue_max_length."
        ),
    )
    task_storage_dir: Path = Path("task-data")
    inference_min_seconds: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    publish_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    publish_retry_delay_seconds: float = Field(
        default=0.5,
        gt=0,
        allow_inf_nan=False,
        description="Backoff for batch_publisher when the soft/hard limit blocks publish.",
    )
    task_max_delivery_attempts: int = Field(
        default=3,
        ge=1,
        description="Failed handler deliveries before reject-to-DLQ (no infinite requeue).",
    )
    task_retry_delay_seconds: float = Field(
        default=1.0,
        gt=0,
        allow_inf_nan=False,
        description="Delay before republishing a failed task for another attempt.",
    )
    worker_health_host: str = "0.0.0.0"
    worker_health_port: int = Field(
        default=8001,
        gt=0,
        lt=65536,
        description="HTTP port for inference-worker /healthz during processing.",
    )
    worker_executor_workers: int = Field(
        default=1,
        ge=1,
        description="Thread pool size; must match broker prefetch_count.",
    )

    @model_validator(mode="after")
    def soft_limit_within_hard_limit(self) -> Self:
        """Keep soft threshold at or below the broker hard ceiling."""
        if self.task_queue_soft_limit > self.task_queue_max_length:
            raise ValueError("task_queue_soft_limit must be <= task_queue_max_length")
        return self


def get_settings() -> Settings:
    """Returns an instance of the application settings.

    Used for Dependency Injection in FastAPI and to fetch the current
    configuration anywhere across the project.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()
