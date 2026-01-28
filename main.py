import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from api.router import router as api_router
from config import settings
from backends.factory import get_backend
from utils.model_download import ensure_tflite_ssd_mobilenet_v1, ModelDownloadError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Light Object Detection API",
    description="A lightweight API for object detection with pluggable backends",
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    """Log backend availability on startup."""
    logger.info("=" * 60)
    logger.info("Light Object Detection API Starting")
    logger.info("=" * 60)
    logger.info(f"Available backends: {settings.AVAILABLE_BACKENDS}")
    logger.info(f"Default backend: {settings.DEFAULT_BACKEND}")

    # Ensure the default TFLite model is present if TFLite is enabled.
    # This keeps the out-of-the-box experience working, while still allowing
    # users to override paths via .env.
    if "tflite" in settings.AVAILABLE_BACKENDS:
        try:
            ensure_result = ensure_tflite_ssd_mobilenet_v1(
                model_path=settings.TFLITE_MODEL_PATH,
                labels_path=settings.TFLITE_LABELS_PATH,
                force=False,
            )
            if ensure_result.did_download:
                logger.info(f"✓ Downloaded default TFLite model: {ensure_result.message}")
        except ModelDownloadError as e:
            logger.error(
                "✗ TFLite model is missing and could not be downloaded. "
                f"Reason: {e}. "
                "Fix: run 'python scripts/download_model.py' or set TFLITE_MODEL_PATH to an existing model."
            )

    # Test each backend
    for backend_name in settings.AVAILABLE_BACKENDS:
        try:
            detector = get_backend(backend_name)
            model_info = detector.get_model_info()
            logger.info(f"✓ {backend_name.upper()} backend: {model_info.get('model_path', 'N/A')}")
        except Exception as e:
            logger.error(f"✗ {backend_name.upper()} backend failed: {str(e)}")

    logger.info("=" * 60)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Light Object Detection API",
        "version": "0.1.0",
        "description": "A lightweight API for object detection with pluggable backends",
        "available_backends": settings.AVAILABLE_BACKENDS,
    }

@app.get("/health")
async def health():
    """Health check endpoint for container orchestrators."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True)
