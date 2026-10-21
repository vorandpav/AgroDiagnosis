"""Verify batch preparation script without external services."""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from src.source import read_tasks

_SCRIPT = Path(__file__).resolve().parents[1] / "task-data" / "scripts" / "prepare_batch.py"
_SPEC = importlib.util.spec_from_file_location("prepare_batch_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
prepare_batch = _MODULE.prepare_batch


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "input"
    source.mkdir()
    return source, tmp_path / "storage"


def test_preparation_generates_unique_ids_and_readable_copies_for_large_batch(
    dirs: tuple[Path, Path],
) -> None:
    source, storage = dirs
    for index in range(120):
        directory = source / str(index % 2)
        directory.mkdir(exist_ok=True)
        Image.new("RGB", (2, 2)).save(directory / f"{index}.PNG")
    manifest, task_count = prepare_batch(source, storage, "demo")
    tasks = list(read_tasks(manifest))
    assert task_count == 120
    assert len({task.task_id for task in tasks}) == 120
    for task in tasks:
        with Image.open(storage / task.image_path) as image:
            assert image.size == (2, 2)
    assert list(read_tasks(manifest)) == tasks


def test_preparation_limits_images_and_preserves_snapshot(dirs: tuple[Path, Path]) -> None:
    source, storage = dirs
    original = source / "a.jpg"
    Image.new("RGB", (2, 2)).save(original)
    Image.new("RGB", (2, 2)).save(source / "b.jpg")
    manifest, task_count = prepare_batch(source, storage, "demo", limit=1)
    task = next(read_tasks(manifest))
    original_bytes = original.read_bytes()
    original.write_bytes(b"changed after preparation")
    assert task_count == 1
    assert (storage / task.image_path).read_bytes() == original_bytes


def test_preparation_does_not_overwrite_existing_batch(dirs: tuple[Path, Path]) -> None:
    source, storage = dirs
    Image.new("RGB", (2, 2)).save(source / "leaf.jpg")
    manifest, _ = prepare_batch(source, storage, "demo")
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_batch(source, storage, "demo")
    assert manifest.read_bytes() == original


def test_new_batch_assigns_new_ids_to_same_images(dirs: tuple[Path, Path]) -> None:
    source, storage = dirs
    Image.new("RGB", (2, 2)).save(source / "leaf.jpg")
    first = next(read_tasks(prepare_batch(source, storage, "demo")[0]))
    second = next(read_tasks(prepare_batch(source, storage, "second")[0]))
    assert first.task_id != second.task_id


def test_preparation_does_not_publish_manifest_when_image_is_corrupt(
    dirs: tuple[Path, Path],
) -> None:
    source, storage = dirs
    Image.new("RGB", (2, 2)).save(source / "a.jpg")
    (source / "b.jpg").write_bytes(b"invalid image")
    with pytest.raises(UnidentifiedImageError):
        prepare_batch(source, storage, "demo")
    assert not (storage / "demo" / "tasks.jsonl").exists()


def test_preparation_rejects_empty_image_selection(dirs: tuple[Path, Path]) -> None:
    source, storage = dirs
    (source / "notes.txt").write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="no supported images"):
        prepare_batch(source, storage, "demo")
    assert not (storage / "demo" / "tasks.jsonl").exists()


def test_preparation_rejects_storage_inside_source(dirs: tuple[Path, Path]) -> None:
    source, _storage = dirs
    with pytest.raises(ValueError, match="outside the source"):
        prepare_batch(source, source / "output", "demo")
    assert not (source / "output").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".", "a/b", "a\\b"])
def test_preparation_rejects_unsafe_batch_names(dirs: tuple[Path, Path], name: str) -> None:
    source, storage = dirs
    with pytest.raises(ValueError, match="Batch name"):
        prepare_batch(source, storage, name)


@pytest.mark.parametrize("limit", [0, -1])
def test_preparation_rejects_nonpositive_limit(dirs: tuple[Path, Path], limit: int) -> None:
    source, storage = dirs
    with pytest.raises(ValueError, match="limit"):
        prepare_batch(source, storage, "demo", limit=limit)
