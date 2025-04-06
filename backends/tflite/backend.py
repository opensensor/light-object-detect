import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite

from backends.base import DetectionBackend
from models.detection import DetectionResult, BoundingBox


class TFLiteBackend(DetectionBackend):
    """TensorFlow Lite detection backend."""
    
    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the TFLite detection backend.
        
        Args:
            model_path: Path to the TFLite model file
            labels_path: Path to the labels file
            confidence_threshold: Default confidence threshold
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        
        # Check if model file exists
        if not os.path.exists(model_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            raise FileNotFoundError(f"TFLite model not found at {model_path}. Please download a model and place it at this location.")
        
        # Load model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        
        # Load labels
        self.labels = self._load_labels(labels_path)
    
    def _load_labels(self, labels_path: str) -> List[str]:
        """
        Load labels from file.
        
        Args:
            labels_path: Path to the labels file
            
        Returns:
            List of label strings
        """
        if not os.path.exists(labels_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            
            # Create a default COCO labels file if it doesn't exist
            default_labels = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]
            
            with open(labels_path, 'w') as f:
                for label in default_labels:
                    f.write(f"{label}\n")
            
            return default_labels
        
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for the model.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as numpy array
        """
        # Resize image to match model input shape
        input_height, input_width = self.input_shape[1], self.input_shape[2]
        resized_image = image.resize((input_width, input_height))
        
        # Convert to numpy array and normalize
        img_array = np.array(resized_image)
        
        # Add batch dimension
        input_data = np.expand_dims(img_array, axis=0)
        
        # Convert to float32 if needed
        if self.input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        
        return input_data
    
    def detect(self, image: Image.Image, confidence_threshold: float = None) -> List[DetectionResult]:
        """
        Detect objects in an image.
        
        Args:
            image: PIL Image to analyze
            confidence_threshold: Minimum confidence score for returned detections
            
        Returns:
            List of DetectionResult objects
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensors
        # The output format depends on the model, but typically:
        # - boxes: [y_min, x_min, y_max, x_max] in normalized coordinates
        # - classes: class indices
        # - scores: confidence scores
        # - num_detections: number of detections
        
        # Extract detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        
        results = []
        
        # Process detections
        for i in range(num_detections):
            if scores[i] >= confidence_threshold:
                # Get class label
                class_id = int(classes[i])
                if class_id < len(self.labels):
                    label = self.labels[class_id]
                else:
                    label = f"class_{class_id}"
                
                # Get bounding box
                # Convert from [y_min, x_min, y_max, x_max] to [x_min, y_min, x_max, y_max]
                y_min, x_min, y_max, x_max = boxes[i]
                
                # Create detection result
                detection = DetectionResult(
                    label=label,
                    confidence=float(scores[i]),
                    bounding_box=BoundingBox(
                        x_min=float(x_min),
                        y_min=float(y_min),
                        x_max=float(x_max),
                        y_max=float(y_max)
                    )
                )
                
                results.append(detection)
        
        return results
    
    def draw_detections(self, image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: PIL Image to draw on
            detections: List of DetectionResult objects
            
        Returns:
            PIL Image with bounding boxes and labels drawn
        """
        # Create a copy of the image to draw on
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Define colors for different classes (cycling through a list)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        
        # Draw each detection
        for i, detection in enumerate(detections):
            # Get pixel coordinates
            box = detection.bounding_box.to_pixel_coords(image.width, image.height)
            x_min, y_min = box["x_min"], box["y_min"]
            x_max, y_max = box["x_max"], box["y_max"]
            
            # Select color based on class (cycling through the colors list)
            color = colors[hash(detection.label) % len(colors)]
            
            # Draw rectangle
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)
            
            # Prepare label text with confidence
            label_text = f"{detection.label}: {detection.confidence:.2f}"
            
            # Calculate text size and position
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:4]
            text_background_coords = [
                (x_min, max(0, y_min - text_height - 2)),
                (x_min + text_width + 4, y_min)
            ]
            
            # Draw text background
            draw.rectangle(text_background_coords, fill=color)
            
            # Draw text
            draw.text(
                (x_min + 2, max(0, y_min - text_height - 2)),
                label_text,
                fill=(255, 255, 255),
                font=font
            )
        
        return draw_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "backend": "tflite",
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "input_shape": list(self.input_shape),
            "num_labels": len(self.labels),
            "default_confidence_threshold": self.confidence_threshold
        }
