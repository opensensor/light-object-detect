from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends, Body
from typing import List, Optional, Dict, Any
import io
import base64
from PIL import Image
import time

from models.detection import DetectionResponse, DetectionResult, ImageResponse
from models.zone import ZoneConfiguration
from backends.factory import get_backend
from utils.image import validate_image, preprocess_image, image_to_bytes
from utils.zones import filter_detections_by_zones, apply_class_filter, apply_size_filter
from config import settings

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    backend: str = Query(settings.DEFAULT_BACKEND, description="Detection backend to use"),
    confidence_threshold: Optional[float] = Query(
        None, description="Confidence threshold for detections (0-1)"
    ),
    return_image: bool = Query(
        False, description="Return the image with bounding boxes drawn"
    ),
    zones: Optional[str] = Query(
        None, description="JSON string of zone configuration for filtering detections"
    ),
    filter_classes: Optional[str] = Query(
        None, description="Comma-separated list of classes to detect (e.g., 'person,car')"
    ),
    min_width: Optional[float] = Query(
        None, description="Minimum normalized width for detections (0-1)", ge=0.0, le=1.0
    ),
    min_height: Optional[float] = Query(
        None, description="Minimum normalized height for detections (0-1)", ge=0.0, le=1.0
    ),
):
    """
    Detect objects in an uploaded image using the specified backend.

    - **file**: Image file to analyze
    - **backend**: Backend to use for detection (default: tflite)
    - **confidence_threshold**: Minimum confidence score for returned detections (0-1)
    - **return_image**: If true, returns the image with bounding boxes drawn
    - **zones**: JSON configuration for detection zones
    - **filter_classes**: Comma-separated list of classes to detect
    - **min_width**: Minimum width filter for detections
    - **min_height**: Minimum height filter for detections
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

    # Apply zone filtering if zones are provided
    if zones:
        try:
            import json
            zone_config = ZoneConfiguration(**json.loads(zones))
            detections = filter_detections_by_zones(detections, zone_config)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid zone configuration: {str(e)}")

    # Apply class filtering if specified
    if filter_classes:
        allowed_classes = [c.strip() for c in filter_classes.split(',')]
        detections = apply_class_filter(detections, allowed_classes=allowed_classes)

    # Apply size filtering if specified
    if min_width is not None or min_height is not None:
        detections = apply_size_filter(
            detections,
            min_width=min_width,
            min_height=min_height
        )

    process_time = time.time() - start_time
    
    # Prepare response
    response = DetectionResponse(
        backend=backend,
        filename=file.filename,
        detections=detections,
        process_time_ms=int(process_time * 1000),
        image_width=image.width,
        image_height=image.height
    )
    
    # If requested, add image with bounding boxes
    if return_image:
        try:
            # Draw bounding boxes on the image
            annotated_image = detector.draw_detections(processed_image, detections)
            
            # Convert image to base64
            img_format = "JPEG"
            if file.filename.lower().endswith(".png"):
                img_format = "PNG"
                
            img_bytes = image_to_bytes(annotated_image, format=img_format)
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # Add to response
            response.image = ImageResponse(
                content_type=f"image/{img_format.lower()}",
                base64_data=img_base64
            )
        except Exception as e:
            # If drawing fails, log but continue without image
            print(f"Error drawing bounding boxes: {str(e)}")
    
    return response


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
