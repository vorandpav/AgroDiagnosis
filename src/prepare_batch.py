"""Prepare a batch: uv run python -m src.prepare_batch IMAGE_DIRECTORY --name demo."""

import argparse
from pathlib import Path
from uuid import uuid4

from src.config import get_settings
from src.schemas import BatchPreparation
from src.services.batch_preparation import prepare_batch


def main() -> None:
    """Prepare a batch with generated task IDs and print its manifest location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--name", default=uuid4().hex, help="New batch directory name")
    parser.add_argument("--limit", type=int, help="Maximum number of images; default: all")
    args = parser.parse_args()
    request = BatchPreparation(
        source_dir=args.source_dir,
        storage_dir=get_settings().task_storage_dir,
        name=args.name,
        limit=args.limit,
    )
    result = prepare_batch(request)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
