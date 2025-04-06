import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from api.router import router as api_router
from config import settings

app = FastAPI(
    title="Light Object Detection API",
    description="A lightweight API for object detection with pluggable backends",
    version="0.1.0",
)

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True)
