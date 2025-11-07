from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from typing import List, Optional, Dict, Any
import io
from PIL import Image
import time

from models.detection import DetectionResponse, DetectionResult
from backends.factory import get_backend
from utils.image import validate_image, preprocess_image
from utils.tracking import MultiStreamTracker
from config import settings

router = APIRouter()

# Global tracker instance
global_tracker = MultiStreamTracker(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
    max_time_since_update=1.0
)


@router.post("/detect-and-track", response_model=DetectionResponse)
async def detect_and_track(
    file: UploadFile = File(...),
    stream_name: str = Query(..., description="Name of the stream for tracking"),
    backend: str = Query(settings.DEFAULT_BACKEND, description="Detection backend to use"),
    confidence_threshold: Optional[float] = Query(
        None, description="Confidence threshold for detections (0-1)"
    ),
):
    """
    Detect objects and track them across frames.
    
    - **file**: Image file to analyze
    - **stream_name**: Name of the stream (required for tracking)
    - **backend**: Backend to use for detection
    - **confidence_threshold**: Minimum confidence score for detections
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
    
    # Apply tracking
    tracked_detections = global_tracker.update(stream_name, detections)
    
    process_time = time.time() - start_time
    
    # Prepare response
    response = DetectionResponse(
        backend=backend,
        filename=file.filename,
        detections=tracked_detections,
        process_time_ms=int(process_time * 1000),
        image_width=image.width,
        image_height=image.height
    )
    
    return response


@router.get("/tracks/{stream_name}", response_model=Dict[str, Any])
async def get_active_tracks(stream_name: str):
    """
    Get active tracks for a stream.
    
    - **stream_name**: Name of the stream
    """
    tracker = global_tracker.get_tracker(stream_name)
    active_tracks = tracker.get_active_tracks()
    
    return {
        "stream_name": stream_name,
        "active_tracks": [
            {
                "track_id": track.track_id,
                "label": track.label,
                "confidence": track.confidence,
                "age": track.age,
                "hits": track.hits,
                "bounding_box": {
                    "x_min": track.bounding_box.x_min,
                    "y_min": track.bounding_box.y_min,
                    "x_max": track.bounding_box.x_max,
                    "y_max": track.bounding_box.y_max
                }
            }
            for track in active_tracks
        ],
        "total_tracks": len(active_tracks)
    }


@router.post("/tracks/{stream_name}/reset")
async def reset_stream_tracker(stream_name: str):
    """
    Reset tracker for a specific stream.
    
    - **stream_name**: Name of the stream
    """
    global_tracker.reset_stream(stream_name)
    return {"message": f"Tracker reset for stream: {stream_name}"}


@router.post("/tracks/reset-all")
async def reset_all_trackers():
    """Reset all stream trackers."""
    global_tracker.reset_all()
    return {"message": "All trackers reset"}

