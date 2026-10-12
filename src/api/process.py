"""Plant-disease inference endpoint."""

import logging

from fastapi import APIRouter, Request, UploadFile

from src.schemas import ProcessResponse
from src.services import inference as inference_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process", response_model=ProcessResponse)
async def process(request: Request, file: UploadFile) -> ProcessResponse:
    """Run inference with the model loaded at application startup.

    Args:
        request: FastAPI HTTP request instance.
        file: Uploaded plant leaf image.

    Returns:
        ProcessResponse: Predicted class, confidence, and model reference.
    """
    image_bytes = await file.read()
    result = inference_service.predict(
        request.app.state.model,
        request.app.state.class_names,
        image_bytes,
        request.app.state.model_ref,
    )
    log.info("Predicted %s (%.3f) via %s", result.label, result.confidence, result.model)
    return result
