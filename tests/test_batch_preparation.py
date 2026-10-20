"""Verify automatic batch preparation without external services."""

from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from src.schemas import BatchPreparation, PreparedBatch
from src.services.batch_preparation import prepare_batch
from src.source import read_tasks


@pytest.fixture
def request_data(tmp_path: Path) -> BatchPreparation:
    source = tmp_path / "input"
    source.mkdir()
    return BatchPreparation(source_dir=source, storage_dir=tmp_path / "storage", name="demo")


def test_preparation_generates_unique_ids_and_readable_copies_for_large_batch(
    request_data: BatchPreparation,
) -> None:
    for index in range(120):
        directory = request_data.source_dir / str(index % 2)
        directory.mkdir(exist_ok=True)
        Image.new("RGB", (2, 2)).save(directory / f"{index}.PNG")
    result = prepare_batch(request_data)
    tasks = list(read_tasks(result.manifest))
    assert result.task_count == 120
    assert len({task.task_id for task in tasks}) == 120
    assert PreparedBatch.model_validate_json(result.model_dump_json()) == result
    for task in tasks:
        with Image.open(request_data.storage_dir / task.image_path) as image:
            assert image.size == (2, 2)
    assert list(read_tasks(result.manifest)) == tasks


def test_preparation_limits_images_and_preserves_snapshot(request_data: BatchPreparation) -> None:
    original = request_data.source_dir / "a.jpg"
    Image.new("RGB", (2, 2)).save(original)
    Image.new("RGB", (2, 2)).save(request_data.source_dir / "b.jpg")
    request_data.limit = 1
    result = prepare_batch(request_data)
    task = next(read_tasks(result.manifest))
    original_bytes = original.read_bytes()
    original.write_bytes(b"changed after preparation")
    assert result.task_count == 1
    assert (request_data.storage_dir / task.image_path).read_bytes() == original_bytes


def test_preparation_does_not_overwrite_existing_batch(request_data: BatchPreparation) -> None:
    Image.new("RGB", (2, 2)).save(request_data.source_dir / "leaf.jpg")
    result = prepare_batch(request_data)
    original = result.manifest.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_batch(request_data)
    assert result.manifest.read_bytes() == original


def test_new_batch_assigns_new_ids_to_same_images(request_data: BatchPreparation) -> None:
    Image.new("RGB", (2, 2)).save(request_data.source_dir / "leaf.jpg")
    first = next(read_tasks(prepare_batch(request_data).manifest))
    request_data.name = "second"
    second = next(read_tasks(prepare_batch(request_data).manifest))
    assert first.task_id != second.task_id


def test_preparation_does_not_publish_manifest_when_image_is_corrupt(
    request_data: BatchPreparation,
) -> None:
    Image.new("RGB", (2, 2)).save(request_data.source_dir / "a.jpg")
    (request_data.source_dir / "b.jpg").write_bytes(b"invalid image")
    with pytest.raises(UnidentifiedImageError):
        prepare_batch(request_data)
    assert not (request_data.storage_dir / "demo" / "tasks.jsonl").exists()


def test_preparation_rejects_empty_image_selection(request_data: BatchPreparation) -> None:
    (request_data.source_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="no supported images"):
        prepare_batch(request_data)
    assert not (request_data.storage_dir / "demo" / "tasks.jsonl").exists()


def test_preparation_rejects_storage_inside_source(request_data: BatchPreparation) -> None:
    request_data.storage_dir = request_data.source_dir / "output"
    with pytest.raises(ValueError, match="outside the source"):
        prepare_batch(request_data)
    assert not request_data.storage_dir.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".", "a/b", "a\\b"])
def test_preparation_contract_rejects_unsafe_batch_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValidationError):
        BatchPreparation(source_dir=tmp_path, storage_dir=tmp_path, name=name)


@pytest.mark.parametrize("limit", [0, -1])
def test_preparation_contract_rejects_nonpositive_limit(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValidationError):
        BatchPreparation(source_dir=tmp_path, storage_dir=tmp_path, name="demo", limit=limit)
