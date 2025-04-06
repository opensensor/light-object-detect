import os
from typing import Tuple, List
from PIL import Image
import io

from config import settings


def validate_image(image: Image.Image, filename: str) -> None:
    """
    Validate that the image meets the requirements.
    
    Args:
        image: PIL Image to validate
        filename: Original filename
        
    Raises:
        ValueError: If the image is invalid
    """
    # Check file extension
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    if ext not in settings.SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}. Supported formats: {settings.SUPPORTED_FORMATS}")
    
    # Check image mode
    if image.mode not in ['RGB', 'RGBA']:
        raise ValueError(f"Unsupported image mode: {image.mode}. Must be RGB or RGBA.")
    
    # Check image dimensions
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"Invalid image dimensions: {image.width}x{image.height}")


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image for detection.
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed PIL Image
    """
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the image on the background
        background.paste(image, mask=image.split()[3])
        image = background
    
    # Resize if image is too large
    max_size = settings.MAX_IMAGE_SIZE
    if image.width > max_size or image.height > max_size:
        # Calculate new dimensions while preserving aspect ratio
        if image.width > image.height:
            new_width = max_size
            new_height = int(image.height * (max_size / image.width))
        else:
            new_height = max_size
            new_width = int(image.width * (max_size / image.height))
        
        # Resize the image
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image


def image_to_bytes(image: Image.Image, format: str = 'JPEG') -> bytes:
    """
    Convert PIL Image to bytes.
    
    Args:
        image: PIL Image to convert
        format: Output format (JPEG, PNG)
        
    Returns:
        Image bytes
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()
