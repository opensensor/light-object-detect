from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional, Tuple


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "Light Object Detection API"

    # Backend settings
    DEFAULT_BACKEND: str = "onnx"
    AVAILABLE_BACKENDS: List[str] = ["onnx", "tflite", "opencv", "edgetpu"]

    # TFLite settings
    TFLITE_MODEL_PATH: str = "backends/tflite/models/ssd_mobilenet_v1.tflite"
    TFLITE_LABELS_PATH: str = "backends/tflite/models/labelmap.txt"
    TFLITE_CONFIDENCE_THRESHOLD: float = 0.5

    # ONNX settings
    ONNX_MODEL_PATH: str = "backends/onnx/models/yolo11n.onnx"
    ONNX_LABELS_PATH: str = "backends/onnx/models/coco.txt"
    ONNX_CONFIDENCE_THRESHOLD: float = 0.5
    ONNX_IOU_THRESHOLD: float = 0.45
    ONNX_MODEL_TYPE: str = "yolo11"

    # OpenCV DNN settings
    OPENCV_MODEL_PATH: str = "backends/opencv/models/yolov4-tiny.weights"
    OPENCV_CONFIG_PATH: str = "backends/opencv/models/yolov4-tiny.cfg"
    OPENCV_LABELS_PATH: str = "backends/opencv/models/coco.names"
    OPENCV_CONFIDENCE_THRESHOLD: float = 0.5
    OPENCV_NMS_THRESHOLD: float = 0.4
    OPENCV_MODEL_TYPE: str = "yolo"
    OPENCV_INPUT_SIZE: Tuple[int, int] = (416, 416)

    # EdgeTPU (Coral) settings
    EDGETPU_MODEL_PATH: str = "backends/edgetpu/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    EDGETPU_LABELS_PATH: str = "backends/edgetpu/models/coco_labels.txt"
    EDGETPU_CONFIDENCE_THRESHOLD: float = 0.5
    EDGETPU_DEVICE: Optional[str] = None  # None=auto, 'usb', 'usb:0', 'pci', 'pci:0'
    EDGETPU_MODEL_TYPE: str = "ssd"  # 'ssd' or 'yolo'
    EDGETPU_IOU_THRESHOLD: float = 0.4  # For YOLO models only
    
    # Image settings
    MAX_IMAGE_SIZE: int = 1024  # Maximum dimension (width or height) in pixels
    SUPPORTED_FORMATS: List[str] = ["jpg", "jpeg", "png"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
