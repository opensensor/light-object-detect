import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2

from backends.base import DetectionBackend
from models.detection import DetectionResult, BoundingBox


class OpenCVBackend(DetectionBackend):
    """OpenCV DNN detection backend supporting various model formats."""
    
    def __init__(self, model_path: str, config_path: str, labels_path: str, 
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4,
                 model_type: str = "yolo", input_size: Tuple[int, int] = (416, 416)):
        """
        Initialize the OpenCV DNN detection backend.
        
        Args:
            model_path: Path to the model file (weights)
            config_path: Path to the model config file (for Darknet/Caffe)
            labels_path: Path to the labels file
            confidence_threshold: Default confidence threshold
            nms_threshold: Non-maximum suppression threshold
            model_type: Type of model (yolo, ssd, faster-rcnn, etc.)
            input_size: Input size for the model (width, height)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model_type = model_type.lower()
        self.input_width, self.input_height = input_size
        
        # Check if model file exists
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please download a model and place it at this location."
            )
        
        # Load the model
        self.net = self._load_model(model_path, config_path)
        
        # Get output layer names
        self.output_layers = self._get_output_layers()
        
        # Load labels
        self.labels = self._load_labels(labels_path)
    
    def _load_model(self, model_path: str, config_path: str) -> cv2.dnn.Net:
        """
        Load the model using OpenCV DNN.
        
        Args:
            model_path: Path to the model file
            config_path: Path to the config file
            
        Returns:
            OpenCV DNN network
        """
        # Determine model format based on file extension
        model_ext = os.path.splitext(model_path)[1].lower()
        
        if model_ext == '.weights':
            # Darknet (YOLO)
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        elif model_ext == '.caffemodel':
            # Caffe
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        elif model_ext == '.pb':
            # TensorFlow
            if os.path.exists(config_path):
                net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            else:
                net = cv2.dnn.readNetFromTensorflow(model_path)
        elif model_ext == '.onnx':
            # ONNX
            net = cv2.dnn.readNetFromONNX(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_ext}")
        
        # Set backend and target
        # Try to use CUDA if available
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            # Fall back to CPU
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return net
    
    def _get_output_layers(self) -> List[str]:
        """
        Get the output layer names.
        
        Returns:
            List of output layer names
        """
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
    
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
        Preprocess image for the model.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Tuple of (blob, scale, original_size)
        """
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            img_cv,
            scalefactor=1/255.0,
            size=(self.input_width, self.input_height),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        
        # Calculate scale for coordinate conversion
        scale_x = image.width / self.input_width
        scale_y = image.height / self.input_height
        
        return blob, (scale_x, scale_y), (image.width, image.height)
    
    def postprocess_yolo(self, outputs: List[np.ndarray], scale: Tuple[float, float],
                        orig_size: Tuple[int, int], confidence_threshold: float) -> List[DetectionResult]:
        """
        Postprocess YOLO outputs.
        
        Args:
            outputs: Model outputs
            scale: Scale factors (scale_x, scale_y)
            orig_size: Original image size (width, height)
            confidence_threshold: Confidence threshold
            
        Returns:
            List of DetectionResult objects
        """
        boxes = []
        confidences = []
        class_ids = []
        
        orig_width, orig_height = orig_size
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # YOLO format: [center_x, center_y, width, height, objectness, class_scores...]
                    center_x = int(detection[0] * orig_width)
                    center_y = int(detection[1] * orig_height)
                    width = int(detection[2] * orig_width)
                    height = int(detection[3] * orig_height)
                    
                    # Calculate top-left corner
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                # Normalize coordinates to [0, 1]
                x_min = max(0.0, x / orig_width)
                y_min = max(0.0, y / orig_height)
                x_max = min(1.0, (x + w) / orig_width)
                y_max = min(1.0, (y + h) / orig_height)
                
                class_id = class_ids[i]
                label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"
                
                detection = DetectionResult(
                    label=label,
                    confidence=confidences[i],
                    bounding_box=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max
                    )
                )
                results.append(detection)
        
        return results

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
        blob, scale, orig_size = self.preprocess_image(image)

        # Set input
        self.net.setInput(blob)

        # Run inference
        outputs = self.net.forward(self.output_layers)

        # Postprocess based on model type
        if self.model_type in ["yolo", "yolov3", "yolov4"]:
            results = self.postprocess_yolo(outputs, scale, orig_size, confidence_threshold)
        else:
            # Default to YOLO postprocessing
            results = self.postprocess_yolo(outputs, scale, orig_size, confidence_threshold)

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
        backend_name = "Unknown"
        target_name = "Unknown"

        try:
            backend_id = self.net.getPreferableBackend()
            target_id = self.net.getPreferableTarget()

            backend_map = {
                cv2.dnn.DNN_BACKEND_OPENCV: "OpenCV",
                cv2.dnn.DNN_BACKEND_CUDA: "CUDA"
            }
            target_map = {
                cv2.dnn.DNN_TARGET_CPU: "CPU",
                cv2.dnn.DNN_TARGET_CUDA: "CUDA"
            }

            backend_name = backend_map.get(backend_id, f"Backend_{backend_id}")
            target_name = target_map.get(target_id, f"Target_{target_id}")
        except:
            pass

        return {
            "backend": "opencv",
            "model_type": self.model_type,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "labels_path": self.labels_path,
            "input_size": [self.input_width, self.input_height],
            "num_labels": len(self.labels),
            "default_confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "dnn_backend": backend_name,
            "dnn_target": target_name
        }

