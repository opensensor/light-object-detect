from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from PIL import Image

from models.detection import DetectionResult


class DetectionBackend(ABC):
    """Base class for all detection backends."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the detection backend."""
        pass
    
    @abstractmethod
    def detect(self, image: Image.Image, confidence_threshold: float = 0.5) -> List[DetectionResult]:
        """
        Detect objects in an image.
        
        Args:
            image: PIL Image to analyze
            confidence_threshold: Minimum confidence score for returned detections
            
        Returns:
            List of DetectionResult objects
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model used by this backend.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    @abstractmethod
    def draw_detections(self, image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: PIL Image to draw on
            detections: List of DetectionResult objects
            
        Returns:
            PIL Image with bounding boxes and labels drawn
        """
        pass
