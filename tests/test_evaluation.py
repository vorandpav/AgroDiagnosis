"""Tests for ZIP dataset evaluation, metrics, and HTTP responses."""

import io
from unittest.mock import MagicMock
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
import torch
from httpx import AsyncClient
from PIL import Image
from pydantic import ValidationError
from torch import nn

from src.app import app
from src.exceptions import EvaluationTrackingError, InvalidDatasetError
from src.schemas import EvaluationMetrics, EvaluationRequest
from src.services import evaluation


class ConstantModel(nn.Module):
    """Always predict the first class."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(batch.shape[0], 2)
        logits[:, 0] = 10
        return logits


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color="green").save(buffer, format="PNG")
    return buffer.getvalue()


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return buffer.getvalue()


def evaluate_archive(archive: bytes) -> evaluation.EvaluationReport:
    return evaluation.evaluate_dataset(
        EvaluationRequest(dataset_name="new-test", archive=archive),
        ConstantModel().eval(),
        ["Apple/healthy", "Cherry/healthy"],
        "plants@champion",
        "7",
    )


def test_evaluation_calculates_metrics_and_preserves_model_version() -> None:
    report = evaluate_archive(
        zip_bytes(
            {
                "Apple/healthy/a.png": image_bytes(),
                "Cherry/healthy/b.png": image_bytes(),
            }
        )
    )
    assert report.metrics.accuracy == 0.5
    assert report.metrics.precision_macro == 0.25
    assert report.metrics.recall_macro == 0.5
    assert report.metrics.f1_macro == pytest.approx(1 / 3)
    assert report.confusion_matrix == [[1, 0], [1, 0]]
    assert report.model_version == "7"
    assert len(report.dataset_sha256) == 64
    assert len(report.predictions[0].sha256) == 64


def test_evaluation_accepts_single_wrapper_folder_and_ignores_mac_metadata() -> None:
    report = evaluate_archive(
        zip_bytes(
            {
                "test/Apple/healthy/a.PNG": image_bytes(),
                "__MACOSX/._a.PNG": b"metadata",
            }
        )
    )
    assert report.metrics.f1_macro == 1
    assert len(report.predictions) == 1
    assert report.labels == ["Apple/healthy"]


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"../Apple/healthy/a.png": b"x"},
        {"/Apple/healthy/a.png": b"x"},
        {"Apple\\healthy\\a.png": b"x"},
        {"Banana/healthy/a.png": b"x"},
        {"Apple/healthy/a.png": b"not an image"},
        {"Apple/healthy/a.txt": b"text"},
        {"a.png": b"x"},
        {"one/Apple/healthy/a.png": b"x", "two/Apple/healthy/b.png": b"x"},
    ],
)
def test_evaluation_rejects_invalid_dataset(files: dict[str, bytes]) -> None:
    with pytest.raises(InvalidDatasetError):
        evaluate_archive(zip_bytes(files))


def test_evaluation_rejects_non_zip() -> None:
    with pytest.raises(InvalidDatasetError, match="valid ZIP"):
        evaluate_archive(b"not a zip")


def test_evaluation_rejects_symlink() -> None:
    buffer = io.BytesIO()
    entry = ZipInfo("Apple/healthy/a.png")
    entry.external_attr = 0o120777 << 16
    with ZipFile(buffer, "w") as archive:
        archive.writestr(entry, b"target")
    with pytest.raises(InvalidDatasetError, match="Links"):
        evaluate_archive(buffer.getvalue())


def test_evaluation_rejects_duplicate_paths() -> None:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("Apple/healthy/a.png", image_bytes())
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("Apple/healthy/a.png", image_bytes())
    with pytest.raises(InvalidDatasetError, match="duplicate"):
        evaluate_archive(buffer.getvalue())


@pytest.mark.parametrize(
    "values",
    [
        {"dataset_name": "", "archive": b"zip"},
        {"dataset_name": "a/b", "archive": b"zip"},
        {"dataset_name": "test", "archive": b""},
    ],
)
def test_evaluation_request_rejects_invalid_fields(values: dict[str, str | bytes]) -> None:
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(values)


def test_evaluation_metrics_reject_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        EvaluationMetrics(accuracy=1.1, precision_macro=0, recall_macro=0, f1_macro=0)


async def test_evaluate_returns_metrics_and_mlflow_run(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "class_names", ["Apple/healthy", "Cherry/healthy"])
    record = MagicMock(return_value="evaluation-run")
    monkeypatch.setattr("src.api.evaluate.record_evaluation", record)
    response = await client.post(
        "/evaluate",
        data={"dataset_name": "fresh-test"},
        files={
            "file": (
                "test.zip",
                zip_bytes({"Apple/healthy/a.png": image_bytes()}),
                "application/zip",
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sample_count"] == 1
    assert body["metrics"]["accuracy"] == 1
    assert body["model_version"] == "1"
    assert body["run_id"] == "evaluation-run"
    assert record.call_args.args[1].dataset_name == "fresh-test"


@pytest.mark.parametrize("archive", [b"", b"invalid"])
async def test_evaluate_returns_400_without_tracking_for_invalid_zip(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, archive: bytes
) -> None:
    record = MagicMock()
    monkeypatch.setattr("src.api.evaluate.record_evaluation", record)
    response = await client.post("/evaluate", files={"file": ("test.zip", archive)})
    assert response.status_code == 400
    assert response.json()["status"] == "invalid_dataset"
    record.assert_not_called()


async def test_evaluate_returns_503_when_model_unavailable(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "model", None)
    response = await client.post("/evaluate", files={"file": ("test.zip", b"zip")})
    assert response.status_code == 503
    assert response.json()["status"] == "model_unavailable"


async def test_evaluate_returns_503_when_tracking_unavailable(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "class_names", ["Apple/healthy", "Cherry/healthy"])
    monkeypatch.setattr(
        "src.api.evaluate.record_evaluation",
        MagicMock(side_effect=EvaluationTrackingError("Evaluation tracking is unavailable")),
    )
    response = await client.post(
        "/evaluate", files={"file": ("test.zip", zip_bytes({"Apple/healthy/a.png": image_bytes()}))}
    )
    assert response.status_code == 503
    assert response.json() == {
        "status": "tracking_unavailable",
        "message": "Evaluation tracking is unavailable",
    }


def test_evaluation_rejects_invalid_compressed_image(monkeypatch: pytest.MonkeyPatch) -> None:
    import zlib

    monkeypatch.setattr(ZipFile, "read", MagicMock(side_effect=zlib.error("corrupt stream")))
    with pytest.raises(InvalidDatasetError, match="Cannot read image"):
        evaluate_archive(zip_bytes({"Apple/healthy/a.png": b"corrupt"}))
