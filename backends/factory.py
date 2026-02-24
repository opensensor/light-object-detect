from typing import Dict, Type

from backends.base import DetectionBackend
from config import settings

# Lazy imports for all optional backends
try:
    from backends.tflite.backend import TFLiteBackend
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

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

try:
    from backends.edgetpu.backend import EdgeTPUBackend
    EDGETPU_AVAILABLE = True
except ImportError:
    EDGETPU_AVAILABLE = False

try:
    from backends.moondream.backend import MoondreamBackend
    MOONDREAM_AVAILABLE = True
except ImportError:
    MOONDREAM_AVAILABLE = False


# Registry of available backends
BACKEND_REGISTRY: Dict[str, Type[DetectionBackend]] = {}

# Instance cache — avoids reloading heavy models (e.g. Moondream 3.85 GB) on
# every request.
_backend_instances: Dict[str, DetectionBackend] = {}

if TFLITE_AVAILABLE:
    BACKEND_REGISTRY["tflite"] = TFLiteBackend

if ONNX_AVAILABLE:
    BACKEND_REGISTRY["onnx"] = ONNXBackend

if OPENCV_AVAILABLE:
    BACKEND_REGISTRY["opencv"] = OpenCVBackend

if EDGETPU_AVAILABLE:
    BACKEND_REGISTRY["edgetpu"] = EdgeTPUBackend

if MOONDREAM_AVAILABLE:
    BACKEND_REGISTRY["moondream"] = MoondreamBackend


def get_backend(backend_name: str) -> DetectionBackend:
    """
    Get a (cached) instance of the specified detection backend.

    Args:
        backend_name: Name of the backend to instantiate

    Returns:
        Instance of the requested backend

    Raises:
        ValueError: If the backend is not available
    """
    if backend_name in _backend_instances:
        return _backend_instances[backend_name]

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Backend '{backend_name}' not found. Available backends: {list(BACKEND_REGISTRY.keys())}")

    backend_class = BACKEND_REGISTRY[backend_name]

    # Initialize backend with appropriate settings
    if backend_name == "tflite":
        instance = backend_class(
            model_path=settings.TFLITE_MODEL_PATH,
            labels_path=settings.TFLITE_LABELS_PATH,
            confidence_threshold=settings.TFLITE_CONFIDENCE_THRESHOLD
        )
    elif backend_name == "onnx":
        instance = backend_class(
            model_path=settings.ONNX_MODEL_PATH,
            labels_path=settings.ONNX_LABELS_PATH,
            confidence_threshold=settings.ONNX_CONFIDENCE_THRESHOLD,
            iou_threshold=settings.ONNX_IOU_THRESHOLD,
            model_type=settings.ONNX_MODEL_TYPE
        )
    elif backend_name == "opencv":
        instance = backend_class(
            model_path=settings.OPENCV_MODEL_PATH,
            config_path=settings.OPENCV_CONFIG_PATH,
            labels_path=settings.OPENCV_LABELS_PATH,
            confidence_threshold=settings.OPENCV_CONFIDENCE_THRESHOLD,
            nms_threshold=settings.OPENCV_NMS_THRESHOLD,
            model_type=settings.OPENCV_MODEL_TYPE,
            input_size=settings.OPENCV_INPUT_SIZE
        )
    elif backend_name == "edgetpu":
        instance = backend_class(
            model_path=settings.EDGETPU_MODEL_PATH,
            labels_path=settings.EDGETPU_LABELS_PATH,
            confidence_threshold=settings.EDGETPU_CONFIDENCE_THRESHOLD,
            device=settings.EDGETPU_DEVICE,
            model_type=settings.EDGETPU_MODEL_TYPE,
            iou_threshold=settings.EDGETPU_IOU_THRESHOLD
        )
    elif backend_name == "moondream":
        instance = backend_class(
            model_name=settings.MOONDREAM_MODEL_NAME,
            revision=settings.MOONDREAM_REVISION,
            device=settings.MOONDREAM_DEVICE,
            default_detect_classes=settings.MOONDREAM_DEFAULT_DETECT_CLASSES,
        )
    else:
        instance = backend_class()

    _backend_instances[backend_name] = instance
    return instance


def register_backend(name: str, backend_class: Type[DetectionBackend]) -> None:
    """
    Register a new backend.
    
    Args:
        name: Name to register the backend under
        backend_class: Backend class to register
    """
    BACKEND_REGISTRY[name] = backend_class
