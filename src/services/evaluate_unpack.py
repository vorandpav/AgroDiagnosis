"""Extract a labelled evaluation ZIP into shared task storage."""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4
from zipfile import BadZipFile, ZipFile

from src.config import Settings
from src.exceptions import InvalidDatasetError
from src.schemas import EvaluationBatchPrepared, EvaluationRequest, InferenceTask
from src.services.evaluation import list_image_entries, read_image_bytes

log = logging.getLogger(__name__)


def prepare_evaluation_batch(
    data: EvaluationRequest,
    settings: Settings,
) -> EvaluationBatchPrepared:
    """Validate the archive, persist images, and build queue tasks."""
    try:
        archive = ZipFile(io.BytesIO(data.archive))
    except BadZipFile as exc:
        raise InvalidDatasetError("File must be a valid ZIP archive") from exc
    batch_id = uuid4()
    root = settings.task_storage_dir.resolve()
    batch_root = (root / "evaluations" / str(batch_id)).resolve()
    if not batch_root.is_relative_to(root):
        raise InvalidDatasetError("Evaluation batch path escapes task storage")
    batch_root.mkdir(parents=True, exist_ok=False)
    try:
        tasks = _write_evaluation_images(archive, batch_id, root)
    except (InvalidDatasetError, OSError):
        _delete_tree(batch_root)
        raise
    return EvaluationBatchPrepared(
        dataset_name=data.dataset_name,
        dataset_sha256=hashlib.sha256(data.archive).hexdigest(),
        tasks=tasks,
    )


def remove_evaluation_batch(prepared: EvaluationBatchPrepared, settings: Settings) -> None:
    """Delete images written for one evaluation batch."""
    _delete_tree(_evaluation_batch_directory(prepared, settings))


def _write_evaluation_images(
    archive: ZipFile,
    batch_id: UUID,
    root: Path,
) -> list[InferenceTask]:
    """Persist archive images under the batch directory and return queue tasks."""
    tasks: list[InferenceTask] = []
    with archive:
        entries = list_image_entries(archive, class_names=None)
        for entry in sorted(entries, key=lambda item: item.filename):
            image_bytes = read_image_bytes(archive, entry)
            relative = Path("evaluations") / str(batch_id) / Path(*entry.filename.split("/"))
            absolute = (root / relative).resolve()
            if not absolute.is_relative_to(root):
                raise InvalidDatasetError("Archive contains an unsafe path")
            absolute.parent.mkdir(parents=True, exist_ok=True)
            absolute.write_bytes(image_bytes)
            tasks.append(InferenceTask(task_id=uuid4(), image_path=relative.as_posix()))
    return tasks


def _evaluation_batch_directory(prepared: EvaluationBatchPrepared, settings: Settings) -> Path:
    """Return the directory prepare_evaluation_batch created for this batch."""
    relative = PurePosixPath(prepared.tasks[0].image_path)
    if len(relative.parts) < 2 or relative.parts[0] != "evaluations":
        raise RuntimeError("Evaluation image path is outside an unpacked batch")
    batch_dir = (settings.task_storage_dir / "evaluations" / relative.parts[1]).resolve()
    storage = (settings.task_storage_dir / "evaluations").resolve()
    if batch_dir == storage or not batch_dir.is_relative_to(storage):
        raise RuntimeError("Evaluation batch path escapes task storage")
    return batch_dir


def _delete_tree(path: Path) -> None:
    """Remove a batch directory, logging when the filesystem refuses."""
    try:
        shutil.rmtree(path)
    except OSError as exc:
        log.warning("Could not delete evaluation batch %s: %s", path, exc)
