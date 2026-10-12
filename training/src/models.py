"""Build MobileNetV3 classifiers for plant-disease fine-tuning."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenet_v3_small(*, num_classes: int, dropout: float) -> nn.Module:
    """Return ImageNet-pretrained MobileNetV3-Small with a custom classifier head.

    Args:
        num_classes: Number of plant-disease classes.
        dropout: Dropout probability before the final linear layer.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(1024, num_classes),
    )
    return model


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick CUDA when available unless an explicit device string is given."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
