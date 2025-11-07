import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort

from backends.base import DetectionBackend
from models.detection import DetectionResult, BoundingBox


class ONNXBackend(DetectionBackend):
    """ONNX Runtime detection backend supporting YOLO and other ONNX models."""
    
    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45, model_type: str = "yolov8"):
        """
        Initialize the ONNX detection backend.
        
        Args:
            model_path: Path to the ONNX model file
            labels_path: Path to the labels file
            confidence_threshold: Default confidence threshold
            iou_threshold: IoU threshold for NMS
            model_type: Type of model (yolov8, yolov5, yolox, etc.)
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_type = model_type.lower()
        
        # Check if model file exists
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                f"Please download a YOLOv8 ONNX model and place it at this location."
            )
        
        # Initialize ONNX Runtime session
        providers = ['CPUExecutionProvider']
        # Try to use CUDA if available
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2] if len(input_shape) > 2 else 640
        self.input_width = input_shape[3] if len(input_shape) > 3 else 640
        
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
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            
            # Create a default COCO labels file
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
    
    def preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO model.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Tuple of (preprocessed image array, scale factor, (pad_w, pad_h))
        """
        # Get original dimensions
        orig_width, orig_height = image.size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(self.input_width / orig_width, self.input_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # Create padded image
        padded_image = Image.new('RGB', (self.input_width, self.input_height), (114, 114, 114))
        pad_w = (self.input_width - new_width) // 2
        pad_h = (self.input_height - new_height) // 2
        padded_image.paste(resized_image, (pad_w, pad_h))
        
        # Convert to numpy array and normalize
        img_array = np.array(padded_image).astype(np.float32)
        
        # Normalize to [0, 1] and convert to CHW format
        img_array = img_array / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, scale, (pad_w, pad_h)
    
    def postprocess_yolov8(self, outputs: List[np.ndarray], scale: float,
                          pad: Tuple[int, int], confidence_threshold: float,
                          orig_width: int, orig_height: int) -> List[DetectionResult]:
        """
        Postprocess YOLOv8 outputs.

        Args:
            outputs: Model outputs
            scale: Scale factor used in preprocessing
            pad: Padding used in preprocessing (pad_w, pad_h)
            confidence_threshold: Confidence threshold
            orig_width: Original image width
            orig_height: Original image height

        Returns:
            List of DetectionResult objects
        """
        # YOLOv8 output shape: (1, 84, 8400) or (1, num_classes+4, num_predictions)
        # Format: [x_center, y_center, width, height, class_scores...]
        output = outputs[0]
        
        # Transpose to (num_predictions, num_classes+4)
        if len(output.shape) == 3:
            output = output[0].T
        
        # Extract boxes and scores
        boxes = output[:, :4]
        scores = output[:, 4:]
        
        # Get class with highest score for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences >= confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert from center format to corner format
        # [x_center, y_center, w, h] -> [x_min, y_min, x_max, y_max]
        x_centers = boxes[:, 0]
        y_centers = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        
        x_mins = x_centers - widths / 2
        y_mins = y_centers - heights / 2
        x_maxs = x_centers + widths / 2
        y_maxs = y_centers + heights / 2
        
        # Apply NMS
        indices = self._nms(
            np.stack([x_mins, y_mins, x_maxs, y_maxs], axis=1),
            confidences,
            self.iou_threshold
        )
        
        results = []
        pad_w, pad_h = pad
        
        for idx in indices:
            # Remove padding and scale back to original image coordinates
            x_min = (x_mins[idx] - pad_w) / scale
            y_min = (y_mins[idx] - pad_h) / scale
            x_max = (x_maxs[idx] - pad_w) / scale
            y_max = (y_maxs[idx] - pad_h) / scale

            # Normalize to [0, 1] using original image dimensions
            x_min_norm = max(0.0, min(1.0, x_min / orig_width))
            y_min_norm = max(0.0, min(1.0, y_min / orig_height))
            x_max_norm = max(0.0, min(1.0, x_max / orig_width))
            y_max_norm = max(0.0, min(1.0, y_max / orig_height))
            
            class_id = int(class_ids[idx])
            label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"
            
            detection = DetectionResult(
                label=label,
                confidence=float(confidences[idx]),
                bounding_box=BoundingBox(
                    x_min=x_min_norm,
                    y_min=y_min_norm,
                    x_max=x_max_norm,
                    y_max=y_max_norm
                )
            )
            results.append(detection)
        
        return results
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression.
        
        Args:
            boxes: Array of boxes in format [x_min, y_min, x_max, y_max]
            scores: Array of confidence scores
            iou_threshold: IoU threshold
            
        Returns:
            List of indices to keep
        """
        if len(boxes) == 0:
            return []
        
        # Sort by score
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            
            # Calculate intersection
            x_min = np.maximum(current_box[0], remaining_boxes[:, 0])
            y_min = np.maximum(current_box[1], remaining_boxes[:, 1])
            x_max = np.minimum(current_box[2], remaining_boxes[:, 2])
            y_max = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
            
            # Calculate union
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                            (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = current_area + remaining_areas - intersection
            
            # Calculate IoU
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU below threshold
            indices = indices[1:][iou <= iou_threshold]
        
        return keep

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

        # Get original image dimensions
        orig_width, orig_height = image.size

        # Preprocess image
        input_data, scale, pad = self.preprocess_image(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # Postprocess based on model type
        if self.model_type in ["yolov8", "yolov5", "yolox"]:
            results = self.postprocess_yolov8(outputs, scale, pad, confidence_threshold, orig_width, orig_height)
        else:
            # Default to YOLOv8 postprocessing
            results = self.postprocess_yolov8(outputs, scale, pad, confidence_threshold, orig_width, orig_height)

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

        # Define colors for different classes
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
        for detection in detections:
            # Get pixel coordinates
            box = detection.bounding_box.to_pixel_coords(image.width, image.height)
            x_min, y_min = box["x_min"], box["y_min"]
            x_max, y_max = box["x_max"], box["y_max"]

            # Select color based on class
            color = colors[hash(detection.label) % len(colors)]

            # Draw rectangle
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)

            # Prepare label text with confidence
            label_text = f"{detection.label}: {detection.confidence:.2f}"

            # Calculate text size and position
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_background_coords = [
                (x_min, max(0, y_min - text_height - 4)),
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
            "backend": "onnx",
            "model_type": self.model_type,
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "input_shape": [1, 3, self.input_height, self.input_width],
            "num_labels": len(self.labels),
            "default_confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "providers": self.session.get_providers()
        }

