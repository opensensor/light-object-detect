from typing import Dict, Type

from backends.base import DetectionBackend
from backends.tflite.backend import TFLiteBackend
from config import settings

# Lazy imports for optional backends
try:
    from backends.onnx.backend import ONNXBackend
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from backends.opencv.backend import OpenCVBackend
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# Registry of available backends
BACKEND_REGISTRY: Dict[str, Type[DetectionBackend]] = {
    "tflite": TFLiteBackend,
}

# Add optional backends if available
if ONNX_AVAILABLE:
    BACKEND_REGISTRY["onnx"] = ONNXBackend

if OPENCV_AVAILABLE:
    BACKEND_REGISTRY["opencv"] = OpenCVBackend


def get_backend(backend_name: str) -> DetectionBackend:
    """
    Get an instance of the specified detection backend.
    
    Args:
        backend_name: Name of the backend to instantiate
        
    Returns:
        Instance of the requested backend
        
    Raises:
        ValueError: If the backend is not available
    """
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Backend '{backend_name}' not found. Available backends: {list(BACKEND_REGISTRY.keys())}")
    
    backend_class = BACKEND_REGISTRY[backend_name]
    
    # Initialize backend with appropriate settings
    if backend_name == "tflite":
        return backend_class(
            model_path=settings.TFLITE_MODEL_PATH,
            labels_path=settings.TFLITE_LABELS_PATH,
            confidence_threshold=settings.TFLITE_CONFIDENCE_THRESHOLD
        )
    elif backend_name == "onnx":
        return backend_class(
            model_path=settings.ONNX_MODEL_PATH,
            labels_path=settings.ONNX_LABELS_PATH,
            confidence_threshold=settings.ONNX_CONFIDENCE_THRESHOLD,
            iou_threshold=settings.ONNX_IOU_THRESHOLD,
            model_type=settings.ONNX_MODEL_TYPE
        )
    elif backend_name == "opencv":
        return backend_class(
            model_path=settings.OPENCV_MODEL_PATH,
            config_path=settings.OPENCV_CONFIG_PATH,
            labels_path=settings.OPENCV_LABELS_PATH,
            confidence_threshold=settings.OPENCV_CONFIDENCE_THRESHOLD,
            nms_threshold=settings.OPENCV_NMS_THRESHOLD,
            model_type=settings.OPENCV_MODEL_TYPE,
            input_size=settings.OPENCV_INPUT_SIZE
        )

    # Default initialization for other backends
    return backend_class()


def register_backend(name: str, backend_class: Type[DetectionBackend]) -> None:
    """
    Register a new backend.
    
    Args:
        name: Name to register the backend under
        backend_class: Backend class to register
    """
    BACKEND_REGISTRY[name] = backend_class
