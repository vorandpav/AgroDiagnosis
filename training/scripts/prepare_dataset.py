"""Copy raw Kaggle splits into nested Plant/Condition folders into DATA_DIR."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from src.settings import get_settings, resolve_path

TEST_PREFIX_TO_CLASS: dict[str, tuple[str, str]] = {
    "AppleCedarRust": ("Apple", "Cedar_apple_rust"),
    "AppleScab": ("Apple", "Apple_scab"),
    "CornCommonRust": ("Corn", "Common_rust"),
    "PotatoEarlyBlight": ("Potato", "Early_blight"),
    "PotatoHealthy": ("Potato", "healthy"),
    "TomatoEarlyBlight": ("Tomato", "Early_blight"),
    "TomatoHealthy": ("Tomato", "healthy"),
    "TomatoYellowCurlVirus": ("Tomato", "Tomato_Yellow_Leaf_Curl_Virus"),
}

_SOURCE_CLASS_RE = re.compile(r"^(.+?)___(.+)$")
_PAREN_RE = re.compile(r"\([^)]*\)")
_TEST_NAME_RE = re.compile(r"^([A-Za-z]+?)\d+\.jpe?g$", re.IGNORECASE)


def _slug(text: str) -> str:
    text = text.replace(",", " ").replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_source_class(name: str) -> tuple[str, str]:
    """``Plant___Condition`` -> nested folder parts."""
    match = _SOURCE_CLASS_RE.match(name)
    if match is None:
        raise ValueError(f"Bad class folder: {name!r}")
    plant = _slug(_PAREN_RE.sub("", match.group(1)))
    condition = _slug(match.group(2).replace("(", "_").replace(")", ""))
    return plant, condition


def parse_test_filename(name: str) -> tuple[str, str]:
    """Flat probe name -> (plant, condition)."""
    match = _TEST_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"Bad test file: {name!r}")
    return TEST_PREFIX_TO_CLASS[match.group(1)]


def _jpg_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".jpg")


def _copy_split(src: Path, dst: Path) -> int:
    n = 0
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        plant, condition = parse_source_class(class_dir.name)
        target = dst / plant / condition
        target.mkdir(parents=True, exist_ok=True)
        for image in _jpg_files(class_dir):
            shutil.copy2(image, target / image.name)
            n += 1
    return n


def _copy_test(src: Path, dst: Path) -> int:
    n = 0
    for image in _jpg_files(src):
        plant, condition = parse_test_filename(image.name)
        target = dst / plant / condition
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target / image.name)
        n += 1
    return n


def main() -> None:
    """Build processed dataset tree."""
    settings = get_settings()
    raw_root = resolve_path(settings.raw_data_dir)
    train_dir = next(p for p in raw_root.rglob("train") if (p.parent / "valid").is_dir())
    valid_dir = train_dir.parent / "valid"
    test_dir = raw_root / "test" / "test"
    out = resolve_path(settings.data_dir)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    n_train = _copy_split(train_dir, out / "train")
    n_val = _copy_split(valid_dir, out / "val")
    n_test = _copy_test(test_dir, out / "test")
    print(f"Wrote {out}")
    print(f"train={n_train} val={n_val} test={n_test}")


if __name__ == "__main__":
    main()
