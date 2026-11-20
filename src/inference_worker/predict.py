"""Run single-image inference with a loaded torch model."""

from __future__ import annotations

import io

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.schemas import InferenceResult

#TODO: Make these configurable via settings or model metadata.

_IMAGE_SIZE = (224, 224)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)


def predict(
    model: nn.Module,
    class_names: list[str],
    image_bytes: bytes,
) -> InferenceResult:
    """Run a single-image forward pass and return the top class.

    Args:
        model: Model loaded at worker startup.
        class_names: Ordered labels matching training class indices.
        image_bytes: Raw uploaded image bytes.

    Returns:
        Predicted label and confidence.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    batch = _EVAL_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(batch)
    probs = torch.softmax(logits, dim=1)[0]
    index = int(torch.argmax(probs).item())
    return InferenceResult(
        label=class_names[index],
        confidence=float(probs[index].item()),
    )
