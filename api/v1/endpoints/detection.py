from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from typing import List, Optional, Dict, Any
import io
from PIL import Image
import time

from models.detection import DetectionResponse, DetectionResult
from backends.factory import get_backend
from utils.image import validate_image, preprocess_image
from config import settings

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    backend: str = Query(settings.DEFAULT_BACKEND, description="Detection backend to use"),
    confidence_threshold: Optional[float] = Query(
        None, description="Confidence threshold for detections (0-1)"
    ),
):
    """
    Detect objects in an uploaded image using the specified backend.
    
    - **file**: Image file to analyze
    - **backend**: Backend to use for detection (default: tflite)
    - **confidence_threshold**: Minimum confidence score for returned detections (0-1)
    """
    # Validate backend
    if backend not in settings.AVAILABLE_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{backend}' not available. Available backends: {settings.AVAILABLE_BACKENDS}"
        )
    
    # Get detection backend
    detector = get_backend(backend)
    
    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(image, file.filename)
        processed_image = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Set confidence threshold
    threshold = confidence_threshold or settings.TFLITE_CONFIDENCE_THRESHOLD
    
    # Perform detection
    start_time = time.time()
    try:
        detections = detector.detect(processed_image, confidence_threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    process_time = time.time() - start_time
    
    # Return results
    return DetectionResponse(
        backend=backend,
        filename=file.filename,
        detections=detections,
        process_time_ms=int(process_time * 1000),
        image_width=image.width,
        image_height=image.height
    )


@router.get("/backends", response_model=Dict[str, Any])
async def list_backends():
    """
    List all available detection backends and their status.
    """
    backends = {}
    for backend_name in settings.AVAILABLE_BACKENDS:
        try:
            detector = get_backend(backend_name)
            backends[backend_name] = {
                "status": "available",
                "model_info": detector.get_model_info()
            }
        except Exception as e:
            backends[backend_name] = {
                "status": "error",
                "error": str(e)
            }
    
    return {
        "default_backend": settings.DEFAULT_BACKEND,
        "backends": backends
    }
