"""Evaluate the loaded model on a labelled ZIP without extracting files to disk."""

from __future__ import annotations

import hashlib
import io
import stat
import zlib
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn

from src.exceptions import InvalidDatasetError
from src.schemas import (
    EvaluationMetrics,
    EvaluationPrediction,
    EvaluationReport,
    EvaluationRequest,
)
from src.services.inference import predict

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _image_entries(archive: ZipFile, class_names: list[str]) -> list[ZipInfo]:
    """Validate paths and labels before reading images."""
    entries: list[ZipInfo] = []
    names: set[str] = set()
    roots: set[str] = set()
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in entry.filename
            or ":" in entry.filename
        ):
            raise InvalidDatasetError("Archive contains an unsafe path")
        if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
            raise InvalidDatasetError("Links and encrypted files are not supported")
        if entry.is_dir() or "__MACOSX" in path.parts or path.name.startswith("."):
            continue
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise InvalidDatasetError("Only JPG, JPEG and PNG images are supported")
        if len(path.parts) not in (3, 4):
            raise InvalidDatasetError(
                "Expected Plant/Condition/image.jpg, optionally inside one folder"
            )
        roots.add(path.parts[0] if len(path.parts) == 4 else "")
        label = "/".join(path.parts[-3:-1])
        if label not in class_names:
            raise InvalidDatasetError(f"Unknown class: {label}")
        if entry.filename in names:
            raise InvalidDatasetError("Archive contains duplicate file paths")
        names.add(entry.filename)
        entries.append(entry)
    if not entries:
        raise InvalidDatasetError("Archive contains no images")
    if len(roots) != 1:
        raise InvalidDatasetError("Images must share the same dataset root")
    return entries


def _read_image(archive: ZipFile, entry: ZipInfo) -> bytes:
    """Read and validate an image while preserving the error cause."""
    try:
        image_bytes = archive.read(entry)
    except (BadZipFile, OSError, RuntimeError, NotImplementedError, zlib.error) as exc:
        raise InvalidDatasetError(f"Cannot read image: {entry.filename}") from exc
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidDatasetError(f"Invalid image: {entry.filename}") from exc
    with image:
        try:
            image.verify()
        except (OSError, SyntaxError, ValueError) as exc:
            raise InvalidDatasetError(f"Invalid image: {entry.filename}") from exc
    return image_bytes


def evaluate_dataset(
    data: EvaluationRequest,
    model: nn.Module,
    class_names: list[str],
    model_ref: str,
    model_version: str,
) -> EvaluationReport:
    """Calculate metrics"""
    try:
        archive = ZipFile(io.BytesIO(data.archive))
    except BadZipFile as exc:
        raise InvalidDatasetError("File must be a valid ZIP archive") from exc
    predictions: list[EvaluationPrediction] = []
    with archive:
        entries = _image_entries(archive, class_names)
        for entry in sorted(entries, key=lambda item: item.filename):
            image_bytes = _read_image(archive, entry)
            try:
                result = predict(model, class_names, image_bytes, model_ref)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise InvalidDatasetError(f"Cannot decode image: {entry.filename}") from exc
            predictions.append(
                EvaluationPrediction(
                    path=entry.filename,
                    sha256=hashlib.sha256(image_bytes).hexdigest(),
                    expected="/".join(PurePosixPath(entry.filename).parts[-3:-1]),
                    predicted=result.label,
                    confidence=result.confidence,
                )
            )
    expected = [row.expected for row in predictions]
    predicted = [row.predicted for row in predictions]
    labels = sorted(set(expected) | set(predicted))
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=labels, average="macro", zero_division=0
    )
    return EvaluationReport(
        dataset_name=data.dataset_name,
        dataset_sha256=hashlib.sha256(data.archive).hexdigest(),
        model=model_ref,
        model_version=model_version,
        metrics=EvaluationMetrics(
            accuracy=float(accuracy_score(expected, predicted)),
            precision_macro=float(precision),
            recall_macro=float(recall),
            f1_macro=float(f1),
        ),
        labels=labels,
        confusion_matrix=confusion_matrix(expected, predicted, labels=labels).tolist(),
        predictions=predictions,
    )
