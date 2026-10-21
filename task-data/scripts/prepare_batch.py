"""Build a JSONL batch from an image directory for the pipeline publisher.

Usage:
    uv run python task-data/scripts/prepare_batch.py IMAGE_DIRECTORY --name demo --limit 100
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from uuid import uuid4

from PIL import Image

from src.schemas import InferenceTask

_BATCH_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_DEFAULT_STORAGE = Path(__file__).resolve().parents[1]


def iter_images(directory: Path) -> Iterator[Path]:
    """Visit supported images in stable order without following symbolic links."""
    for root, directories, filenames in directory.walk():
        directories.sort()
        for filename in sorted(filenames):
            image = root / filename
            if image.suffix.lower() in _IMAGE_SUFFIXES and not image.is_symlink():
                yield image


def prepare_batch(
    source_dir: Path,
    storage_dir: Path,
    name: str,
    limit: int | None = None,
) -> tuple[Path, int]:
    """Copy images into storage and write tasks.jsonl; return manifest path and count."""
    if not _BATCH_NAME.fullmatch(name):
        raise ValueError(
            "Batch name must start with an alphanumeric character "
            "and contain only letters, digits, '_' or '-'"
        )
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    source = source_dir.resolve()
    storage = storage_dir.resolve()
    if storage.is_relative_to(source):
        raise ValueError("Task storage must be outside the source directory")

    batch = storage / name
    batch.mkdir(parents=True, exist_ok=False)
    images = batch / "images"
    images.mkdir()
    pending_manifest = batch / "tasks.jsonl.tmp"
    task_count = 0
    with pending_manifest.open("x", encoding="utf-8") as manifest:
        for original in islice(iter_images(source), limit):
            task_id = uuid4()
            copied = images / f"{task_id}{original.suffix.lower()}"
            shutil.copyfile(original, copied)
            with Image.open(copied) as image:
                image.verify()
            task = InferenceTask(
                task_id=task_id,
                image_path=copied.relative_to(storage).as_posix(),
            )
            manifest.write(task.model_dump_json() + "\n")
            task_count += 1
    if task_count == 0:
        raise ValueError("Source directory contains no supported images")
    completed_manifest = batch / "tasks.jsonl"
    pending_manifest.rename(completed_manifest)
    return completed_manifest, task_count


def main() -> None:
    """Parse CLI args, prepare the batch, print the manifest path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--name", default=uuid4().hex, help="New batch directory name")
    parser.add_argument("--limit", type=int, default=None, help="Max images; default: all")
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=_DEFAULT_STORAGE,
        help="Batch root (default: task-data/)",
    )
    args = parser.parse_args()
    manifest, task_count = prepare_batch(args.source_dir, args.storage_dir, args.name, args.limit)
    print(f"{task_count} tasks -> {manifest}")


if __name__ == "__main__":
    main()
