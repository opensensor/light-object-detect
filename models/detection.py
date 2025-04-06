from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


class BoundingBox(BaseModel):
    """Bounding box coordinates for a detected object."""
    
    x_min: float = Field(..., description="Normalized left coordinate (0-1)")
    y_min: float = Field(..., description="Normalized top coordinate (0-1)")
    x_max: float = Field(..., description="Normalized right coordinate (0-1)")
    y_max: float = Field(..., description="Normalized bottom coordinate (0-1)")
    
    def to_pixel_coords(self, width: int, height: int) -> Dict[str, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return {
            "x_min": int(self.x_min * width),
            "y_min": int(self.y_min * height),
            "x_max": int(self.x_max * width),
            "y_max": int(self.y_max * height)
        }


class DetectionResult(BaseModel):
    """Result of a single object detection."""
    
    label: str = Field(..., description="Class label of the detected object")
    confidence: float = Field(..., description="Confidence score (0-1)")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    
    class Config:
        schema_extra = {
            "example": {
                "label": "person",
                "confidence": 0.92,
                "bounding_box": {
                    "x_min": 0.1,
                    "y_min": 0.2,
                    "x_max": 0.3,
                    "y_max": 0.4
                }
            }
        }


class ImageResponse(BaseModel):
    """Image data response model."""
    
    content_type: str = Field(..., description="Image content type (e.g., 'image/jpeg')")
    base64_data: str = Field(..., description="Base64 encoded image data")


class DetectionResponse(BaseModel):
    """Response model for object detection API."""
    
    backend: str = Field(..., description="Backend used for detection")
    filename: str = Field(..., description="Original filename")
    detections: List[DetectionResult] = Field(default_factory=list, description="List of detected objects")
    process_time_ms: int = Field(..., description="Processing time in milliseconds")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    image: Optional[ImageResponse] = Field(None, description="Image with bounding boxes (if requested)")
    
    class Config:
        schema_extra = {
            "example": {
                "backend": "tflite",
                "filename": "example.jpg",
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.92,
                        "bounding_box": {
                            "x_min": 0.1,
                            "y_min": 0.2,
                            "x_max": 0.3,
                            "y_max": 0.4
                        }
                    }
                ],
                "process_time_ms": 150,
                "image_width": 640,
                "image_height": 480,
                "image": None
            }
        }
