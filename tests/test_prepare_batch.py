"""Verify batch preparation script without app/queue imports."""

import importlib.util
import json
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image, UnidentifiedImageError

_SCRIPT = Path(__file__).resolve().parents[1] / "task-data" / "scripts" / "prepare_batch.py"
_SPEC = importlib.util.spec_from_file_location("prepare_batch_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
prepare_batch = _MODULE.prepare_batch


def _read_records(manifest: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]


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
    records = _read_records(manifest)
    assert task_count == 120
    assert len({record["id"] for record in records}) == 120
    for record in records:
        UUID(record["id"])
        assert set(record) == {"id", "path"}
        with Image.open(storage / record["path"]) as image:
            assert image.size == (2, 2)


def test_preparation_limits_images_and_preserves_snapshot(dirs: tuple[Path, Path]) -> None:
    source, storage = dirs
    original = source / "a.jpg"
    Image.new("RGB", (2, 2)).save(original)
    Image.new("RGB", (2, 2)).save(source / "b.jpg")
    manifest, task_count = prepare_batch(source, storage, "demo", limit=1)
    record = _read_records(manifest)[0]
    original_bytes = original.read_bytes()
    original.write_bytes(b"changed after preparation")
    assert task_count == 1
    assert (storage / record["path"]).read_bytes() == original_bytes


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
    first = _read_records(prepare_batch(source, storage, "demo")[0])[0]
    second = _read_records(prepare_batch(source, storage, "second")[0])[0]
    assert first["id"] != second["id"]


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
