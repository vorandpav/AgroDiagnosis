"""Plant-disease inference endpoint."""

import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from src.schemas import InferenceResult
from src.services import inference as inference_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process", response_model=InferenceResult)
async def process(request: Request, file: UploadFile) -> InferenceResult:
    """Run inference with the model loaded at application startup.

    Args:
        request: FastAPI HTTP request instance.
        file: Uploaded plant leaf image.

    Returns:
        InferenceResult: Predicted class, confidence, and model reference.

    Raises:
        HTTPException: 503 when the model was not loaded at startup.
    """
    if request.app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is unavailable",
        )

    image_bytes = await file.read()
    result = inference_service.predict(
        request.app.state.model,
        request.app.state.class_names,
        image_bytes,
        request.app.state.model_ref,
    )
    log.info("Predicted %s (%.3f) via %s", result.label, result.confidence, result.model)
    return result
