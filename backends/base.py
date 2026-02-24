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

    def describe(self, image: Image.Image, length: str = "normal") -> str:
        """
        Generate a natural language description of the image.

        Args:
            image: PIL Image to describe
            length: Caption length - 'short' or 'normal'

        Returns:
            Natural language description string

        Raises:
            NotImplementedError: If this backend does not support image description
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support image description")

    def query(self, image: Image.Image, question: str) -> str:
        """
        Answer a free-form question about the image.

        Args:
            image: PIL Image to query
            question: Natural language question about the image

        Returns:
            Answer string

        Raises:
            NotImplementedError: If this backend does not support visual Q&A
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support visual Q&A")
