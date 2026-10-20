"""Application API Pydantic contracts and response schemas."""

from enum import StrEnum

from pydantic import BaseModel, Field


class HelloResponse(BaseModel):
    """Response schema for the root endpoint."""

    message: str = Field(default="Hello, world!", description="Welcome greeting message.")
    app: str = Field(description="Name of the application.")
    version: str = Field(description="Application semantic version.")


class LivenessResponse(BaseModel):
    """Response model for the lightweight liveness probe."""

    status: str = Field(default="ok", description="Process liveness indicator.")


class VersionResponse(BaseModel):
    """Response model for the application version endpoint."""

    version: str = Field(description="Application version.")


class HealthStatus(StrEnum):
    """Health status of an individual application dependency."""

    healthy = "healthy"
    unavailable = "unavailable"


class ReportStatus(StrEnum):
    """Overall status of the application in the health report."""

    ok = "ok"
    degraded = "degraded"


class DependencyHealth(BaseModel):
    """Health status and details of a single dependency."""

    status: HealthStatus
    detail: str | None = Field(default=None, description="Status detail or error message.")


class HealthReport(BaseModel):
    """Summary health report encompassing all application dependencies."""

    status: ReportStatus
    dependencies: dict[str, DependencyHealth]


class ProcessResponse(BaseModel):
    """Response schema for plant-disease inference."""

    label: str = Field(description="Predicted plant/disease class.")
    confidence: float = Field(description="Softmax probability of the predicted class.")
    model: str = Field(description="Loaded Model Registry reference (name@alias).")


class EvaluationRequest(BaseModel):
    """Input schema for an uploaded labelled test dataset."""

    dataset_name: str = Field(min_length=1, max_length=100, pattern=r"^[\w .-]+$")
    archive: bytes = Field(min_length=1)


class EvaluationMetrics(BaseModel):
    """Metrics over classes present in predictions or ground truth."""

    accuracy: float = Field(ge=0, le=1)
    precision_macro: float = Field(ge=0, le=1)
    recall_macro: float = Field(ge=0, le=1)
    f1_macro: float = Field(ge=0, le=1)


class EvaluationPrediction(BaseModel):
    """Ground truth and prediction for a single image."""

    path: str
    sha256: str
    expected: str
    predicted: str
    confidence: float = Field(ge=0, le=1)


class EvaluationReport(BaseModel):
    """Evaluation report for MLflow without the original images."""

    dataset_name: str
    dataset_sha256: str
    model: str
    model_version: str
    metrics: EvaluationMetrics
    labels: list[str]
    confusion_matrix: list[list[int]]
    predictions: list[EvaluationPrediction]


class EvaluationResponse(BaseModel):
    """Response schema containing evaluation metrics and tracking identifiers."""

    dataset_name: str
    dataset_sha256: str
    sample_count: int = Field(gt=0)
    model: str
    model_version: str
    metrics: EvaluationMetrics
    run_id: str


class EvaluationError(BaseModel):
    """Response schema containing a stable error code and a readable message."""

    status: str
    message: str
