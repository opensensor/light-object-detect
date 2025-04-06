from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Optional

from api.v1.endpoints import detection
from config import settings

router = APIRouter()

# Include API version 1 endpoints
router.include_router(
    detection.router,
    prefix=settings.API_V1_STR,
    tags=["detection"]
)
