from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "Light Object Detection API"
    
    # Backend settings
    DEFAULT_BACKEND: str = "tflite"
    AVAILABLE_BACKENDS: List[str] = ["tflite"]
    
    # TFLite settings
    TFLITE_MODEL_PATH: str = "backends/tflite/models/ssd_mobilenet_v1.tflite"
    TFLITE_LABELS_PATH: str = "backends/tflite/models/labelmap.txt"
    TFLITE_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Image settings
    MAX_IMAGE_SIZE: int = 1024  # Maximum dimension (width or height) in pixels
    SUPPORTED_FORMATS: List[str] = ["jpg", "jpeg", "png"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
