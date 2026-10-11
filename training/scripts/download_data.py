from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

TRAINING_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRAINING_ROOT / "data"
TARGET_DIR = DATA_DIR / "plant-diseases"


def main() -> None:
    """Install the dataset from Kagglehub."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset via kagglehub")
    cache_path = Path(kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset"))
    print(f"Cached at: {cache_path}")

    if TARGET_DIR.exists():
        print(f"Removing old {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)

    print(f"Copying to {TARGET_DIR}")
    shutil.copytree(cache_path, TARGET_DIR)
    print("Done")


if __name__ == "__main__":
    main()
