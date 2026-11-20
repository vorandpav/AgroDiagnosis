"""Tests for ZIP unpack, aggregate metrics, and HTTP evaluation responses."""

import io
from pathlib import Path
from unittest.mock import MagicMock, Mock
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from httpx import AsyncClient
from PIL import Image
from pydantic import ValidationError

from src.app import app
from src.config import Settings
from src.exceptions import EvaluationTrackingError, InvalidDatasetError
from src.schemas import (
    EvaluationMetrics,
    EvaluationReport,
    EvaluationRequest,
    InferenceResult,
    TaskResult,
)
from src.services.evaluate_unpack import prepare_evaluation_batch
from src.services.evaluation_aggregate import build_evaluation_report
from src.services.evaluation_flow import run_evaluation_flow


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


def evaluate_archive(archive: bytes, tmp_path: Path) -> EvaluationReport:
    """Unpack like production, then aggregate with a constant Apple/healthy worker."""
    settings = Settings(task_storage_dir=tmp_path)
    prepared = prepare_evaluation_batch(
        EvaluationRequest(dataset_name="new-test", archive=archive),
        settings,
    )
    results = {
        task.task_id: TaskResult(
            task_id=task.task_id,
            prediction=InferenceResult(
                label="Apple/healthy",
                confidence=1.0,
            ),
            model_ref="plants@champion",
            model_version="7",
            processing_seconds=0.01,
        )
        for task in prepared.tasks
    }
    return build_evaluation_report(
        dataset_name=prepared.dataset_name,
        dataset_sha256=prepared.dataset_sha256,
        model_ref="plants@champion",
        model_version="7",
        tasks=prepared.tasks,
        results=results,
        settings=settings,
    )


def test_evaluation_calculates_metrics_and_preserves_model_version(tmp_path: Path) -> None:
    report = evaluate_archive(
        zip_bytes(
            {
                "Apple/healthy/a.png": image_bytes(),
                "Cherry/healthy/b.png": image_bytes(),
            }
        ),
        tmp_path,
    )
    assert report.metrics.accuracy == 0.5
    assert report.metrics.precision_macro == 0.25
    assert report.metrics.recall_macro == 0.5
    assert report.metrics.f1_macro == pytest.approx(1 / 3)
    assert report.confusion_matrix == [[1, 0], [1, 0]]
    assert report.model_version == "7"
    assert len(report.dataset_sha256) == 64
    assert len(report.predictions[0].sha256) == 64


def test_evaluation_accepts_single_wrapper_folder_and_ignores_mac_metadata(
    tmp_path: Path,
) -> None:
    report = evaluate_archive(
        zip_bytes(
            {
                "test/Apple/healthy/a.PNG": image_bytes(),
                "__MACOSX/._a.PNG": b"metadata",
            }
        ),
        tmp_path,
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
        {"Apple/healthy/a.png": b"not an image"},
        {"Apple/healthy/a.txt": b"text"},
        {"a.png": b"x"},
        {"one/Apple/healthy/a.png": b"x", "two/Apple/healthy/b.png": b"x"},
    ],
)
def test_evaluation_rejects_invalid_dataset(files: dict[str, bytes], tmp_path: Path) -> None:
    with pytest.raises(InvalidDatasetError):
        prepare_evaluation_batch(
            EvaluationRequest(dataset_name="new-test", archive=zip_bytes(files)),
            Settings(task_storage_dir=tmp_path),
        )


def test_evaluation_rejects_non_zip(tmp_path: Path) -> None:
    with pytest.raises(InvalidDatasetError, match="valid ZIP"):
        prepare_evaluation_batch(
            EvaluationRequest(dataset_name="new-test", archive=b"not a zip"),
            Settings(task_storage_dir=tmp_path),
        )
    assert not (tmp_path / "evaluations").exists()


def test_evaluation_rejects_symlink(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    entry = ZipInfo("Apple/healthy/a.png")
    entry.external_attr = 0o120777 << 16
    with ZipFile(buffer, "w") as archive:
        archive.writestr(entry, b"target")
    with pytest.raises(InvalidDatasetError, match="Links"):
        prepare_evaluation_batch(
            EvaluationRequest(dataset_name="new-test", archive=buffer.getvalue()),
            Settings(task_storage_dir=tmp_path),
        )


def test_evaluation_rejects_duplicate_paths(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("Apple/healthy/a.png", image_bytes())
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("Apple/healthy/a.png", image_bytes())
    with pytest.raises(InvalidDatasetError, match="duplicate"):
        prepare_evaluation_batch(
            EvaluationRequest(dataset_name="new-test", archive=buffer.getvalue()),
            Settings(task_storage_dir=tmp_path),
        )


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
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app.state.settings = app.state.settings.model_copy(update={"task_storage_dir": tmp_path})
    worker_results: dict[UUID, TaskResult] = {}

    async def fake_batch_process_evaluate(tasks, broker, pool, settings) -> dict[UUID, TaskResult]:
        del broker, pool, settings
        for task in tasks:
            worker_results[task.task_id] = TaskResult(
                task_id=task.task_id,
                prediction=InferenceResult(
                    label="Apple/healthy",
                    confidence=1.0,
                ),
                model_ref="plants@champion",
                model_version="1",
                processing_seconds=0.01,
            )
        return worker_results

    record = MagicMock(return_value="evaluation-run")
    monkeypatch.setattr(
        "src.services.evaluation_flow.batch_process_evaluate",
        fake_batch_process_evaluate,
    )
    monkeypatch.setattr("src.services.evaluation_flow.record_evaluation", record)
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
    assert body["tracking"]["status"] == "recorded"
    assert record.call_args.args[1].dataset_name == "fresh-test"


@pytest.mark.parametrize("archive", [b"", b"invalid"])
async def test_evaluate_returns_400_without_tracking_for_invalid_zip(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, archive: bytes
) -> None:
    record = MagicMock()
    monkeypatch.setattr("src.services.evaluation_flow.record_evaluation", record)
    response = await client.post("/evaluate", files={"file": ("test.zip", archive)})
    assert response.status_code == 400
    assert response.json()["status"] == "invalid_dataset"
    record.assert_not_called()


async def test_evaluate_returns_503_when_tracking_unavailable(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app.state.settings = app.state.settings.model_copy(update={"task_storage_dir": tmp_path})
    worker_results: dict[UUID, TaskResult] = {}

    async def fake_batch_process_evaluate(tasks, broker, pool, settings) -> dict[UUID, TaskResult]:
        del broker, pool, settings
        for task in tasks:
            worker_results[task.task_id] = TaskResult(
                task_id=task.task_id,
                prediction=InferenceResult(
                    label="Apple/healthy",
                    confidence=1.0,
                ),
                model_ref="plants@champion",
                model_version="1",
                processing_seconds=0.01,
            )
        return worker_results

    monkeypatch.setattr(
        "src.services.evaluation_flow.batch_process_evaluate",
        fake_batch_process_evaluate,
    )
    monkeypatch.setattr(
        "src.services.evaluation_flow.record_evaluation",
        MagicMock(side_effect=EvaluationTrackingError("Evaluation tracking is unavailable")),
    )
    response = await client.post(
        "/evaluate", files={"file": ("test.zip", zip_bytes({"Apple/healthy/a.png": image_bytes()}))}
    )
    assert response.status_code == 503
    body = response.json()
    assert body["sample_count"] == 1
    assert body["metrics"]["accuracy"] == 1
    assert body["run_id"] is None
    assert body["tracking"] == {
        "status": "unavailable",
        "message": "Evaluation tracking is unavailable",
    }


def test_evaluation_rejects_invalid_compressed_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import zlib

    monkeypatch.setattr(ZipFile, "read", MagicMock(side_effect=zlib.error("corrupt stream")))
    with pytest.raises(InvalidDatasetError, match="Cannot read image"):
        prepare_evaluation_batch(
            EvaluationRequest(
                dataset_name="new-test",
                archive=zip_bytes({"Apple/healthy/a.png": b"corrupt"}),
            ),
            Settings(task_storage_dir=tmp_path),
        )


def _stored_result(task_id: UUID) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        prediction=InferenceResult(
            label="Apple/healthy",
            confidence=1.0,
        ),
        model_ref="plants@champion",
        model_version="1",
        processing_seconds=0.01,
    )


def test_invalid_dataset_does_not_leave_extracted_files(tmp_path: Path) -> None:
    with pytest.raises(InvalidDatasetError):
        prepare_evaluation_batch(
            EvaluationRequest(
                dataset_name="new-test",
                archive=zip_bytes({"notes.txt": b"text"}),
            ),
            Settings(task_storage_dir=tmp_path),
        )
    evaluations = tmp_path / "evaluations"
    assert not evaluations.exists() or not any(evaluations.iterdir())


async def test_evaluation_flow_removes_batch_directory_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(task_storage_dir=tmp_path)
    batch_dirs: list[Path] = []

    async def fake_batch(tasks, broker, pool, settings_arg) -> dict[UUID, TaskResult]:
        del broker, pool, settings_arg
        relative = Path(tasks[0].image_path)
        batch_dir = tmp_path / relative.parts[0] / relative.parts[1]
        assert batch_dir.is_dir()
        assert any(batch_dir.rglob("*.png"))
        batch_dirs.append(batch_dir)
        return {task.task_id: _stored_result(task.task_id) for task in tasks}

    monkeypatch.setattr("src.services.evaluation_flow.batch_process_evaluate", fake_batch)
    monkeypatch.setattr(
        "src.services.evaluation_flow.record_evaluation",
        lambda _client, _report: "run-1",
    )
    result = await run_evaluation_flow(
        EvaluationRequest(
            dataset_name="fresh-test",
            archive=zip_bytes({"Apple/healthy/a.png": image_bytes()}),
        ),
        broker=Mock(),
        pool=Mock(),
        settings=settings,
    )
    assert result.status_code == 200
    assert result.response is not None
    assert result.response.run_id == "run-1"
    assert batch_dirs
    assert not batch_dirs[0].exists()


async def test_evaluation_flow_removes_batch_directory_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(task_storage_dir=tmp_path)
    batch_dirs: list[Path] = []

    async def fake_batch(tasks, broker, pool, settings_arg) -> None:
        del broker, pool, settings_arg
        relative = Path(tasks[0].image_path)
        batch_dir = tmp_path / relative.parts[0] / relative.parts[1]
        assert any(batch_dir.rglob("*.png"))
        batch_dirs.append(batch_dir)
        raise TimeoutError("Evaluation timed out after 120 seconds")

    monkeypatch.setattr("src.services.evaluation_flow.batch_process_evaluate", fake_batch)
    result = await run_evaluation_flow(
        EvaluationRequest(
            dataset_name="fresh-test",
            archive=zip_bytes({"Apple/healthy/a.png": image_bytes()}),
        ),
        broker=Mock(),
        pool=Mock(),
        settings=settings,
    )
    assert result.status_code == 503
    assert result.error is not None
    assert result.error.status == "evaluation_timeout"
    assert batch_dirs
    assert not batch_dirs[0].exists()
