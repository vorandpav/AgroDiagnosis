"""Download the Kaggle plant-diseases archive into RAW_DATA_ROOT."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from src.settings import get_settings, resolve_path


def main() -> None:
    """Install the dataset from Kagglehub."""
    target = resolve_path(get_settings().raw_data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset via kagglehub")
    cache_path = Path(kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset"))
    print(f"Cached at: {cache_path}")

    if target.exists():
        print(f"Removing old {target}")
        shutil.rmtree(target)

    print(f"Copying to {target}")
    shutil.copytree(cache_path, target)
    print("Done")


if __name__ == "__main__":
    main()
