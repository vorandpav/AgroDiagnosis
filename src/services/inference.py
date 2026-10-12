"""Load the champion model from MLflow and run image inference."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import mlflow
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.config import Settings
from src.schemas import ProcessResponse

log = logging.getLogger(__name__)

_CLASS_NAMES_PATH = Path(__file__).resolve().parent.parent / "class_names.json"
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


def load_model(settings: Settings) -> tuple[nn.Module, list[str]]:
    """Load the Registry model once and class names for label mapping.

    Args:
        settings: Application settings with MLflow URI and model identity.

    Returns:
        Tuple of eval-mode model and ordered class name list.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    try:
        model = mlflow.pytorch.load_model(uri)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {uri}") from exc

    model.eval()
    class_names: list[str] = json.loads(_CLASS_NAMES_PATH.read_text(encoding="utf-8-sig"))
    log.info("Loaded model %s (%d classes)", uri, len(class_names))
    return model, class_names


def predict(
    model: nn.Module,
    class_names: list[str],
    image_bytes: bytes,
    loaded_model: str,
) -> ProcessResponse:
    """Run a single-image forward pass and return the top class.

    Args:
        model: Model loaded at startup.
        class_names: Ordered labels matching training class indices.
        image_bytes: Raw uploaded image bytes.
        loaded_model: Registry reference string for the response.

    Returns:
        Predicted label, confidence, and model reference.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    batch = _EVAL_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(batch)
    probs = torch.softmax(logits, dim=1)[0]
    index = int(torch.argmax(probs).item())
    return ProcessResponse(
        label=class_names[index],
        confidence=float(probs[index].item()),
        model=loaded_model,
    )
