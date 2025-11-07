from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional, Tuple


class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "Light Object Detection API"
    
    # Backend settings
    DEFAULT_BACKEND: str = "tflite"
    AVAILABLE_BACKENDS: List[str] = ["tflite", "onnx", "opencv"]

    # TFLite settings
    TFLITE_MODEL_PATH: str = "backends/tflite/models/ssd_mobilenet_v1.tflite"
    TFLITE_LABELS_PATH: str = "backends/tflite/models/labelmap.txt"
    TFLITE_CONFIDENCE_THRESHOLD: float = 0.5

    # ONNX settings
    ONNX_MODEL_PATH: str = "backends/onnx/models/yolov8n.onnx"
    ONNX_LABELS_PATH: str = "backends/onnx/models/coco.txt"
    ONNX_CONFIDENCE_THRESHOLD: float = 0.5
    ONNX_IOU_THRESHOLD: float = 0.45
    ONNX_MODEL_TYPE: str = "yolov8"

    # OpenCV DNN settings
    OPENCV_MODEL_PATH: str = "backends/opencv/models/yolov4-tiny.weights"
    OPENCV_CONFIG_PATH: str = "backends/opencv/models/yolov4-tiny.cfg"
    OPENCV_LABELS_PATH: str = "backends/opencv/models/coco.names"
    OPENCV_CONFIDENCE_THRESHOLD: float = 0.5
    OPENCV_NMS_THRESHOLD: float = 0.4
    OPENCV_MODEL_TYPE: str = "yolo"
    OPENCV_INPUT_SIZE: Tuple[int, int] = (416, 416)
    
    # Image settings
    MAX_IMAGE_SIZE: int = 1024  # Maximum dimension (width or height) in pixels
    SUPPORTED_FORMATS: List[str] = ["jpg", "jpeg", "png"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
