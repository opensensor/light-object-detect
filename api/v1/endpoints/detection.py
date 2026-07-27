from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends, Body
from typing import List, Optional, Dict, Any
import asyncio
import io
import base64
from PIL import Image
import time
import logging

from models.detection import (
    DetectionResponse, DetectionResult, BoundingBox, ImageResponse,
    DescribeResponse, QueryResponse,
)
from models.zone import ZoneConfiguration
from backends.factory import get_backend, BACKEND_REGISTRY
from utils.image import validate_image, preprocess_image, image_to_bytes
from utils.zones import filter_detections_by_zones, apply_class_filter, apply_size_filter
from utils.tiling import (
    MAX_TILES, TileDetection, is_truncated, map_tile_box_to_frame,
    merge_tile_detections, plan_tiles, select_tiles, sweep_cycles,
    tile_grid_overflowed,
)
from config import settings

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()


async def _detect_tiled(
    detector,
    image,
    threshold: float,
    budget: int,
    min_object_px: int,
    overlap: float,
    tile_period: float,
    deadline_s: float,
    start_time: float,
) -> tuple:
    """Run up to ``budget`` inferences over a rotating tile grid and merge the results.

    Returns ``(detections, tiles_run, plan_size)``. Detections come back in
    frame-normalized coordinates, so the existing zone/class/size filters apply
    downstream unchanged.
    """
    frame_w, frame_h = image.size
    model_w = getattr(detector, "input_width", 0) or 640
    model_h = getattr(detector, "input_height", 0) or 640

    plan = plan_tiles(frame_w, frame_h, model_w, model_h, min_object_px, overlap)
    if tile_grid_overflowed(frame_w, frame_h, model_w, model_h, min_object_px, overlap):
        logger.warning(
            "Tiling: grid for %dx%d at min_object_px=%d exceeds the %d-tile cap and was "
            "truncated — raise min_object_px or the sweep will not cover the frame",
            frame_w, frame_h, min_object_px, MAX_TILES,
        )

    # Cursor is keyed to when the request began, so a request that happens to span a
    # second boundary still picks one consistent tile set.
    selected = select_tiles(plan, budget, tile_period, start_time)
    logger.info(
        "Tiling: %dx%d frame, %d-tile grid, running %d this cycle (full sweep %d cycles)",
        frame_w, frame_h, len(plan), len(selected), sweep_cycles(len(plan), budget),
    )

    collected = []
    tiles_run = 0
    for tile in selected:
        # Always run tile 0 so a slow host degrades to current behaviour rather than
        # returning nothing. Only then does the deadline gate subsequent tiles.
        if tiles_run and (time.time() - start_time) >= deadline_s:
            logger.warning(
                "Tiling: deadline of %.1fs reached, stopping after %d/%d tiles — "
                "results are partial for this cycle",
                deadline_s, tiles_run, len(selected),
            )
            break

        crop = image if tile.full_frame else image.crop(tile.box)
        tile_detections = await asyncio.to_thread(
            detector.detect, crop, confidence_threshold=threshold
        )
        tiles_run += 1

        for det in tile_detections:
            b = det.bounding_box
            tile_box = (b.x_min, b.y_min, b.x_max, b.y_max)
            collected.append(
                TileDetection(
                    label=det.label,
                    confidence=det.confidence,
                    box=map_tile_box_to_frame(tile_box, tile, frame_w, frame_h),
                    tile_index=tile.index,
                    truncated=(
                        not tile.full_frame
                        and is_truncated(tile_box, tile, frame_w, frame_h)
                    ),
                )
            )

    merged = merge_tile_detections(
        collected, iou_threshold=settings.TILE_IOU_THRESHOLD
    )
    logger.info(
        "Tiling: %d raw detections across %d tiles merged to %d",
        len(collected), tiles_run, len(merged),
    )

    detections = [
        DetectionResult(
            label=m.label,
            confidence=m.confidence,
            bounding_box=BoundingBox(
                x_min=m.box[0], y_min=m.box[1], x_max=m.box[2], y_max=m.box[3]
            ),
        )
        for m in merged
    ]
    return detections, tiles_run, len(plan)


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
    tiles: int = Query(
        settings.TILE_BUDGET,
        description="Inferences per request (T). 1 disables tiling and uses the whole frame.",
        ge=1, le=MAX_TILES,
    ),
    min_object_px: int = Query(
        settings.TILE_MIN_OBJECT_PX,
        description="Smallest object to resolve, in source pixels. Drives the tile size.",
        ge=1,
    ),
    tile_overlap: float = Query(
        settings.TILE_OVERLAP,
        description="Overlap between adjacent tiles as a fraction of tile size",
        ge=0.0, lt=1.0,
    ),
    tile_period: float = Query(
        settings.TILE_PERIOD_SECONDS,
        description="Rotation cursor period in seconds; match the stream's detection_interval",
        gt=0.0,
    ),
    tile_deadline_s: float = Query(
        settings.TILE_DEADLINE_SECONDS,
        description="Stop issuing tiles once this many seconds have elapsed",
        gt=0.0,
    ),
    stream: Optional[str] = Query(
        None, description="Stream name, logged for correlation (used by change-priority later)"
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
    - **tiles**: Fixed inference budget T per request. With T>1 the frame is split into
      overlapping crops sized so a `min_object_px` object survives the scale down to
      model input; tile 0 is always the whole frame and the remaining T-1 rotate over
      the grid on a clock-derived cursor. Cost is exactly T inferences regardless of
      scene content.
    - **min_object_px**: Smallest object to resolve, in source pixels
    - **tile_overlap**: Overlap between adjacent tiles (0-1)
    - **tile_period**: Rotation period in seconds; set to the stream's detection_interval
    - **tile_deadline_s**: Elapsed-time budget after which remaining tiles are skipped
    - **stream**: Stream name, logged for correlation
    """
    logger.info(
        f"Detection request: backend={backend}, confidence={confidence_threshold}, "
        f"filename={file.filename}, stream={stream}, tiles={tiles}"
    )
    # Remap tflite to edgetpu if edgetpu is set as default backend
    # This allows LightNVR (which has no edgetpu option) to use the Coral TPU
    if backend == "tflite" and settings.DEFAULT_BACKEND == "edgetpu":
        backend = "edgetpu"
        logger.info("Remapping tflite backend request to edgetpu")

    # Validate backend
    if backend not in settings.AVAILABLE_BACKENDS:
        logger.error(f"Invalid backend requested: {backend}")
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{backend}' not available. Available backends: {settings.AVAILABLE_BACKENDS}"
        )

    # Get detection backend
    try:
        detector = get_backend(backend)
        logger.debug(f"Backend {backend} initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize backend {backend}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backend initialization error: {str(e)}")

    # Read and validate image
    try:
        contents = await file.read()
        logger.debug(f"Read {len(contents)} bytes from uploaded file")
        image = Image.open(io.BytesIO(contents))
        logger.debug(f"Image opened: size={image.size}, mode={image.mode}")
        validate_image(image, file.filename)
        # Full resolution survives to the backend: it letterboxes to model input
        # itself, so an intermediate downscale here would only cost small objects.
        processed_image = preprocess_image(
            image, max_size=settings.MAX_DETECTION_IMAGE_SIZE
        )
        logger.debug(f"Image preprocessed: size={processed_image.size}")
    except Exception as e:
        logger.error(f"Image validation/processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Set confidence threshold — use the backend-specific default
    if backend == "onnx":
        default_threshold = settings.ONNX_CONFIDENCE_THRESHOLD
    elif backend == "tflite":
        default_threshold = settings.TFLITE_CONFIDENCE_THRESHOLD
    else:
        default_threshold = 0.5
    threshold = confidence_threshold or default_threshold
    logger.debug(f"Using confidence threshold: {threshold}")

    # Perform detection (run in thread pool to avoid blocking the event loop
    # so health probes can still respond during long-running inference).
    start_time = time.time()
    try:
        if tiles > 1:
            detections, tiles_run, plan_size = await _detect_tiled(
                detector=detector,
                image=processed_image,
                threshold=threshold,
                budget=tiles,
                min_object_px=min_object_px,
                overlap=tile_overlap,
                tile_period=tile_period,
                deadline_s=tile_deadline_s,
                start_time=start_time,
            )
        else:
            # Untiled path, unchanged: one inference over the whole frame.
            tiles_run, plan_size = 1, 1
            detections = await asyncio.to_thread(
                detector.detect, processed_image, confidence_threshold=threshold
            )
        detection_time = time.time() - start_time
        logger.info(
            f"Detection completed: {len(detections)} objects found in "
            f"{detection_time*1000:.1f}ms ({tiles_run}/{plan_size} tiles)"
        )

        # Log detected objects
        if detections:
            class_counts = {}
            for det in detections:
                class_counts[det.label] = class_counts.get(det.label, 0) + 1
            logger.info(f"Detected objects: {dict(class_counts)}")
            for det in detections:
                b = det.bounding_box
                logger.info(f"  → {det.label} ({det.confidence:.2f}): [{b.x_min:.3f},{b.y_min:.3f},{b.x_max:.3f},{b.y_max:.3f}]")
        else:
            logger.info("No objects detected")
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

    # Apply zone filtering if zones are provided
    if zones:
        try:
            import json
            zone_config = ZoneConfiguration(**json.loads(zones))
            initial_count = len(detections)
            detections = filter_detections_by_zones(detections, zone_config)
            logger.info(f"Zone filtering: {initial_count} -> {len(detections)} detections")
        except Exception as e:
            logger.error(f"Zone filtering failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid zone configuration: {str(e)}")

    # Apply class filtering if specified
    if filter_classes:
        allowed_classes = [c.strip() for c in filter_classes.split(',')]
        initial_count = len(detections)
        detections = apply_class_filter(detections, allowed_classes=allowed_classes)
        logger.info(f"Class filtering ({filter_classes}): {initial_count} -> {len(detections)} detections")

    # Apply size filtering if specified
    if min_width is not None or min_height is not None:
        initial_count = len(detections)
        detections = apply_size_filter(
            detections,
            min_width=min_width,
            min_height=min_height
        )
        logger.info(f"Size filtering (w>={min_width}, h>={min_height}): {initial_count} -> {len(detections)} detections")

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

    logger.info(f"Response prepared: {len(detections)} detections, {process_time*1000:.1f}ms total")

    # If requested, add image with bounding boxes
    if return_image:
        try:
            logger.debug("Drawing bounding boxes on image")
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
            logger.debug(f"Annotated image added to response ({len(img_base64)} bytes base64)")
        except Exception as e:
            # If drawing fails, log but continue without image
            logger.error(f"Error drawing bounding boxes: {str(e)}", exc_info=True)

    return response


@router.post("/describe", response_model=DescribeResponse)
async def describe_image(
    file: UploadFile = File(...),
    backend: str = Query("moondream", description="Backend to use for description"),
    length: str = Query("normal", description="Caption length: 'short', 'normal', or 'long'"),
    return_image: bool = Query(False, description="Ignored for describe endpoint (accepted for compatibility)"),
):
    """
    Generate a natural language description of an uploaded image.

    - **file**: Image file to describe
    - **backend**: Backend to use (must support description, e.g. moondream)
    - **length**: Caption length - 'short', 'normal', or 'long'
    """
    logger.info(f"Describe request: backend={backend}, length={length}, filename={file.filename}")

    if backend not in settings.AVAILABLE_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{backend}' not available. Available backends: {settings.AVAILABLE_BACKENDS}"
        )

    try:
        detector = get_backend(backend)
    except Exception as e:
        logger.error(f"Failed to initialize backend {backend}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backend initialization error: {str(e)}")

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(image, file.filename)
        processed_image = preprocess_image(image)
    except Exception as e:
        logger.error(f"Image validation/processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    start_time = time.time()
    try:
        description = await asyncio.to_thread(
            detector.describe, processed_image, length=length
        )
        process_time = time.time() - start_time
        logger.info(f"Describe completed in {process_time*1000:.1f}ms")
        logger.info(f"Description result: {description}")
    except NotImplementedError:
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{backend}' does not support image description"
        )
    except Exception as e:
        logger.error(f"Describe failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Description error: {str(e)}")

    return DescribeResponse(
        backend=backend,
        filename=file.filename,
        description=description,
        process_time_ms=int(process_time * 1000),
        image_width=image.width,
        image_height=image.height,
    )


@router.post("/query", response_model=QueryResponse)
async def query_image(
    file: UploadFile = File(...),
    question: str = Query(..., description="Question to ask about the image"),
    backend: str = Query("moondream", description="Backend to use for visual Q&A"),
    return_image: bool = Query(False, description="Ignored for query endpoint (accepted for compatibility)"),
):
    """
    Ask a natural language question about an uploaded image.

    - **file**: Image file to query
    - **question**: Natural language question about the image
    - **backend**: Backend to use (must support visual Q&A, e.g. moondream)
    """
    logger.info(f"Query request: backend={backend}, question={question!r}, filename={file.filename}")

    if backend not in settings.AVAILABLE_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{backend}' not available. Available backends: {settings.AVAILABLE_BACKENDS}"
        )

    try:
        detector = get_backend(backend)
    except Exception as e:
        logger.error(f"Failed to initialize backend {backend}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backend initialization error: {str(e)}")

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(image, file.filename)
        processed_image = preprocess_image(image)
    except Exception as e:
        logger.error(f"Image validation/processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    start_time = time.time()
    try:
        answer = await asyncio.to_thread(
            detector.query, processed_image, question=question
        )
        process_time = time.time() - start_time
        logger.info(f"Query completed in {process_time*1000:.1f}ms")
        logger.info(f"Query answer: {answer}")
    except NotImplementedError:
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{backend}' does not support visual Q&A"
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

    return QueryResponse(
        backend=backend,
        filename=file.filename,
        question=question,
        answer=answer,
        process_time_ms=int(process_time * 1000),
        image_width=image.width,
        image_height=image.height,
    )


@router.get("/backends", response_model=Dict[str, Any])
async def list_backends():
    """
    List all available detection backends and their status.
    """
    backends = {}
    for backend_name in settings.AVAILABLE_BACKENDS:
        if backend_name not in BACKEND_REGISTRY:
            backends[backend_name] = {
                "status": "unavailable",
                "error": f"Backend '{backend_name}' dependencies not installed"
            }
            continue
        try:
            detector = get_backend(backend_name)
            backends[backend_name] = {
                "status": "available",
                "model_info": detector.get_model_info()
            }
        except Exception as e:
            logger.warning(f"Backend {backend_name} failed to initialize: {e}")
            backends[backend_name] = {
                "status": "error",
                "error": str(e)
            }

    return {
        "default_backend": settings.DEFAULT_BACKEND,
        "backends": backends
    }
