"""Plant-disease image dataset and DataLoader helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SampleRef:
    """One image on disk with its class index."""

    path: Path
    label: int
    class_name: str
    size_bytes: int
    split: str


@dataclass(frozen=True)
class SplitBundle:
    """Train/val/test loaders plus manifest fields for MLflow."""

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    class_names: list[str]
    manifest_paths: list[str]
    manifest_classes: list[str]
    manifest_splits: list[str]
    manifest_sizes: list[int]
    n_train: int
    n_val: int
    n_test: int


class PlantImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map sample refs through a torchvision transform."""

    def __init__(
        self,
        samples: Sequence[SampleRef],
        transform: transforms.Compose,
    ) -> None:
        self._samples = list(samples)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[index]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
        return self._transform(image), torch.tensor(sample.label, dtype=torch.long)


def load_class_names(data_dir: Path) -> list[str]:
    """Sorted ``Plant/Condition`` labels from ``data_dir/train``."""
    train_dir = data_dir / "train"
    return [
        f"{plant.name}/{condition.name}"
        for plant in sorted(train_dir.iterdir())
        if plant.is_dir()
        for condition in sorted(plant.iterdir())
        if condition.is_dir()
    ]


def scan_split(
    split_dir: Path,
    *,
    split_name: str,
    class_names: Sequence[str],
    max_per_class: int | None,
    random_state: int,
) -> list[SampleRef]:
    """Scan ``split/<Plant>/<Condition>/`` and optionally cap per class."""
    label_by_class = {name: idx for idx, name in enumerate(class_names)}
    rng = random.Random(random_state)
    samples: list[SampleRef] = []

    for class_name in class_names:
        plant, condition = class_name.split("/", maxsplit=1)
        class_dir = split_dir / plant / condition
        if not class_dir.is_dir():
            continue
        files = [p for p in class_dir.iterdir() if p.suffix.lower() == ".jpg"]
        rng.shuffle(files)
        if max_per_class is not None:
            files = files[:max_per_class]
        for path in files:
            samples.append(
                SampleRef(
                    path=path,
                    label=label_by_class[class_name],
                    class_name=class_name,
                    size_bytes=path.stat().st_size,
                    split=split_name,
                )
            )
    return samples


def _train_transform(image_size: tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transform(image_size: tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _make_loader(
    samples: Sequence[SampleRef],
    *,
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    kwargs: dict[str, bool | int] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        PlantImageDataset(samples, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def build_split_bundle(
    data_dir: Path,
    *,
    train_per_class: int | None,
    val_per_class: int | None,
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    random_state: int,
    pin_memory: bool,
) -> SplitBundle:
    """Build loaders from processed ``train/`` ``val/`` ``test/`` trees."""
    class_names = load_class_names(data_dir)
    train_samples = scan_split(
        data_dir / "train",
        split_name="train",
        class_names=class_names,
        max_per_class=train_per_class,
        random_state=random_state,
    )
    val_samples = scan_split(
        data_dir / "val",
        split_name="val",
        class_names=class_names,
        max_per_class=val_per_class,
        random_state=random_state,
    )
    test_samples = scan_split(
        data_dir / "test",
        split_name="test",
        class_names=class_names,
        max_per_class=None,
        random_state=random_state,
    )
    all_samples = train_samples + val_samples + test_samples
    eval_tf = _eval_transform(image_size)

    return SplitBundle(
        train_loader=_make_loader(
            train_samples,
            transform=_train_transform(image_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        val_loader=_make_loader(
            val_samples,
            transform=eval_tf,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        test_loader=_make_loader(
            test_samples,
            transform=eval_tf,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        class_names=list(class_names),
        manifest_paths=[str(s.path.relative_to(data_dir)).replace("\\", "/") for s in all_samples],
        manifest_classes=[s.class_name for s in all_samples],
        manifest_splits=[s.split for s in all_samples],
        manifest_sizes=[s.size_bytes for s in all_samples],
        n_train=len(train_samples),
        n_val=len(val_samples),
        n_test=len(test_samples),
    )
