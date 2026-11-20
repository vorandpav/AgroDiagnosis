"""Validate and read labelled evaluation ZIP archives."""

from __future__ import annotations

import io
import stat
import zlib
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

from src.exceptions import InvalidDatasetError

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_image_entries(archive: ZipFile, class_names: list[str] | None = None) -> list[ZipInfo]:
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
        if class_names is not None and label not in class_names:
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


def read_image_bytes(archive: ZipFile, entry: ZipInfo) -> bytes:
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
