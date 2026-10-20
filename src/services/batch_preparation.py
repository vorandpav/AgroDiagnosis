"""Prepare immutable image copies and a streaming JSONL manifest."""

import shutil
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from uuid import uuid4

from PIL import Image

from src.schemas import BatchPreparation, InferenceTask, PreparedBatch


def iter_images(directory: Path) -> Iterator[Path]:
    """Visit supported images in stable order without following symbolic links."""
    for root, directories, filenames in directory.walk():
        directories.sort()
        for filename in sorted(filenames):
            image = root / filename
            if image.suffix.lower() in {".jpg", ".jpeg", ".png"} and not image.is_symlink():
                yield image


def prepare_batch(request: BatchPreparation) -> PreparedBatch:
    """Copy images and expose the manifest only after preparation succeeds."""
    source = request.source_dir.resolve()
    storage = request.storage_dir.resolve()
    if storage.is_relative_to(source):
        raise ValueError("Task storage must be outside the source directory")
    batch = storage / request.name
    batch.mkdir(parents=True, exist_ok=False)
    images = batch / "images"
    images.mkdir()
    pending_manifest = batch / "tasks.jsonl.tmp"
    task_count = 0
    with pending_manifest.open("x", encoding="utf-8") as manifest:
        for original in islice(iter_images(source), request.limit):
            task_id = uuid4()
            copied = images / f"{task_id}{original.suffix.lower()}"
            shutil.copyfile(original, copied)
            with Image.open(copied) as image:
                image.verify()
            task = InferenceTask(task_id=task_id, image_path=copied.relative_to(storage).as_posix())
            manifest.write(task.model_dump_json() + "\n")
            task_count += 1
    if task_count == 0:
        raise ValueError("Source directory contains no supported images")
    completed_manifest = batch / "tasks.jsonl"
    pending_manifest.rename(completed_manifest)
    return PreparedBatch(manifest=completed_manifest, task_count=task_count)
