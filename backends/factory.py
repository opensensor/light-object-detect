from typing import Dict, Type

from backends.base import DetectionBackend
from backends.tflite.backend import TFLiteBackend
from config import settings


# Registry of available backends
BACKEND_REGISTRY: Dict[str, Type[DetectionBackend]] = {
    "tflite": TFLiteBackend,
}


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
