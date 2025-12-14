"""
Coral EdgeTPU detection backend for light-object-detect.

Supports both SSD and YOLO models compiled for EdgeTPU.
Requires libedgetpu runtime and pycoral/tflite_runtime.

Based on Frigate's edgetpu_tfl.py implementation.
"""
import os
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from backends.base import DetectionBackend
from models.detection import DetectionResult, BoundingBox

logger = logging.getLogger(__name__)

# Try to import TFLite interpreter with EdgeTPU delegate support
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    TFLITE_SOURCE = "tflite_runtime"
except ModuleNotFoundError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate
        TFLITE_SOURCE = "tensorflow"
    except ModuleNotFoundError:
        raise ImportError(
            "No TFLite interpreter found. Please install one of:\n"
            "  pip install tflite-runtime  (recommended for EdgeTPU)\n"
            "  pip install tensorflow\n"
            "Also ensure libedgetpu is installed: apt install libedgetpu1-std"
        )


class EdgeTPUBackend(DetectionBackend):
    """
    Coral EdgeTPU detection backend.
    
    Supports:
    - SSD MobileNet models (standard Coral models)
    - YOLO models compiled for EdgeTPU
    
    Requires EdgeTPU-compiled .tflite models (usually named *_edgetpu.tflite)
    """
    
    # Model type constants
    MODEL_TYPE_SSD = "ssd"
    MODEL_TYPE_YOLO = "yolo"
    
    def __init__(
        self,
        model_path: str,
        labels_path: str,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        model_type: str = "ssd",
        iou_threshold: float = 0.4,
    ):
        """
        Initialize the EdgeTPU detection backend.
        
        Args:
            model_path: Path to the EdgeTPU-compiled .tflite model
            labels_path: Path to the labels file
            confidence_threshold: Default confidence threshold (0.0-1.0)
            device: EdgeTPU device specifier (None for auto, or 'usb', 'usb:0', 'pci', 'pci:0')
            model_type: Type of model - 'ssd' or 'yolo'
            iou_threshold: IoU threshold for NMS (YOLO models only)
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model_type = model_type.lower()
        self.iou_threshold = iou_threshold
        
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"EdgeTPU model not found at {model_path}. "
                f"Please download an EdgeTPU-compiled model."
            )
        
        # Validate model is EdgeTPU compiled
        if not model_path.endswith('.tflite'):
            raise ValueError(
                "EdgeTPU models must be .tflite files. "
                "Use models compiled with the Edge TPU Compiler."
            )
        
        # Load EdgeTPU delegate and interpreter
        self._load_interpreter()
        
        # Load labels
        self.labels = self._load_labels(labels_path)
        
        # Setup model-specific parameters
        self._setup_model_params()
        
        logger.info(
            f"EdgeTPU backend initialized: model={os.path.basename(model_path)}, "
            f"type={self.model_type}, device={self.device or 'auto'}, "
            f"input_shape={self.input_shape}, tflite_source={TFLITE_SOURCE}"
        )
    
    def _load_interpreter(self):
        """Load the TFLite interpreter with EdgeTPU delegate."""
        device_config = {}
        if self.device is not None:
            device_config = {"device": self.device}
        
        try:
            device_type = device_config.get("device", "auto")
            logger.info(f"Attempting to load EdgeTPU as {device_type}")
            
            edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
            logger.info("EdgeTPU found and loaded successfully")
            
            self.interpreter = Interpreter(
                model_path=self.model_path,
                experimental_delegates=[edge_tpu_delegate],
            )
        except ValueError as e:
            if "edgetpu" in str(e).lower() or "delegate" in str(e).lower():
                logger.error(
                    "No EdgeTPU was detected. Ensure:\n"
                    "  1. libedgetpu is installed: apt install libedgetpu1-std\n"
                    "  2. EdgeTPU device is connected and detected\n"
                    "  3. For USB: check 'lsusb | grep Google'\n"
                    "  4. For PCIe/M.2: check 'ls /dev/apex_0'"
                )
            raise
        
        self.interpreter.allocate_tensors()
        
        # Get tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        # Check if model requires int8 input
        self.requires_int8 = self.input_details[0]['dtype'] == np.int8

    def _setup_model_params(self):
        """Setup model-specific parameters based on model type."""
        self.max_detections = 20
        self.min_score = self.confidence_threshold

        if self.model_type == self.MODEL_TYPE_YOLO:
            if not CV2_AVAILABLE:
                raise ImportError(
                    "OpenCV (cv2) is required for YOLO model NMS. "
                    "Install with: pip install opencv-python-headless"
                )

            # YOLO-specific setup
            self.reg_max = 16  # Standard YOLO DFL channels
            self.min_logit_value = np.log(self.min_score / (1 - self.min_score))
            self._generate_anchors_and_strides()
            self.project = np.arange(self.reg_max, dtype=np.float32)

            # Determine output tensor indices for YOLO
            self._setup_yolo_outputs()

        elif self.model_type == self.MODEL_TYPE_SSD:
            # SSD models typically have 4 outputs: boxes, classes, scores, count
            self.output_boxes_index = None
            self.output_count_index = None
            self.output_class_ids_index = None
            self.output_scores_index = None

            # Will be determined at first inference
            self._ssd_indices_determined = False

    def _generate_anchors_and_strides(self):
        """Generate anchor points and strides for YOLO DFL decoding."""
        all_anchors = []
        all_strides = []
        strides = (8, 16, 32)  # YOLO detection head strides

        for stride in strides:
            feat_h = self.input_height // stride
            feat_w = self.input_width // stride

            grid_y, grid_x = np.meshgrid(
                np.arange(feat_h, dtype=np.float32),
                np.arange(feat_w, dtype=np.float32),
                indexing="ij",
            )

            grid_coords = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
            anchor_points = grid_coords + 0.5

            all_anchors.append(anchor_points)
            all_strides.append(np.full((feat_h * feat_w, 1), stride, dtype=np.float32))

        self.anchors = np.concatenate(all_anchors, axis=0)
        self.anchor_strides = np.concatenate(all_strides, axis=0)

    def _setup_yolo_outputs(self):
        """Determine YOLO output tensor indices and quantization parameters."""
        output_boxes_index = None
        output_classes_index = None

        for i, details in enumerate(self.output_details):
            shape = details['shape']
            # YOLO outputs have shape (B, N, C) where N is anchor count
            if len(shape) == 3:
                if shape[2] == 64:  # Box coordinates (4 * 16 DFL bins)
                    output_boxes_index = i
                elif shape[2] > 1:  # Class scores (num_classes)
                    output_classes_index = i

        if output_boxes_index is None or output_classes_index is None:
            logger.warning("Could not auto-detect YOLO output tensors, using defaults")
            output_classes_index = 0
            output_boxes_index = 1

        # Store indices and quantization parameters
        scores_details = self.output_details[output_classes_index]
        self.scores_tensor_index = scores_details['index']
        self.scores_scale, self.scores_zero_point = scores_details.get('quantization', (1.0, 0))

        # Calculate quantized min score threshold
        if self.scores_scale > 0:
            self.min_score_quantized = int(
                (self.min_logit_value / self.scores_scale) + self.scores_zero_point
            )
            self.logit_shift = max(0, math.ceil((128 + self.scores_zero_point) * self.scores_scale)) + 1
        else:
            self.min_score_quantized = 0
            self.logit_shift = 0

        boxes_details = self.output_details[output_boxes_index]
        self.boxes_tensor_index = boxes_details['index']
        self.boxes_scale, self.boxes_zero_point = boxes_details.get('quantization', (1.0, 0))

    def _load_labels(self, labels_path: str) -> List[str]:
        """Load labels from file."""
        if not os.path.exists(labels_path):
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)

            # Create default COCO labels
            default_labels = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush"
            ]

            with open(labels_path, 'w') as f:
                for label in default_labels:
                    f.write(f"{label}\n")

            return default_labels

        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _preprocess(self, tensor_input: np.ndarray) -> np.ndarray:
        """Preprocess input tensor for EdgeTPU (handle int8 shift if needed)."""
        if self.requires_int8:
            # Shift uint8 [0,255] to int8 [-128,127]
            tensor_input = np.bitwise_xor(tensor_input, 128).view(np.int8)
        return tensor_input

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess PIL Image for the model.

        Args:
            image: PIL Image to preprocess

        Returns:
            Preprocessed numpy array ready for inference
        """
        # Resize to model input size
        resized = image.resize((self.input_width, self.input_height), Image.BILINEAR)

        # Convert to numpy array
        img_array = np.array(resized, dtype=np.uint8)

        # Ensure 3 channels (RGB)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Apply int8 preprocessing if needed
        img_array = self._preprocess(img_array)

        return img_array

    def detect(self, image: Image.Image, confidence_threshold: float = None) -> List[DetectionResult]:
        """
        Detect objects in an image using EdgeTPU.

        Args:
            image: PIL Image to analyze
            confidence_threshold: Minimum confidence score (overrides default)

        Returns:
            List of DetectionResult objects
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Store original dimensions for coordinate conversion
        orig_width, orig_height = image.size

        # Preprocess image
        input_data = self.preprocess_image(image)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Postprocess based on model type
        if self.model_type == self.MODEL_TYPE_YOLO:
            return self._postprocess_yolo(confidence_threshold, orig_width, orig_height)
        else:
            return self._postprocess_ssd(confidence_threshold, orig_width, orig_height)

    def _postprocess_ssd(
        self, confidence_threshold: float, orig_width: int, orig_height: int
    ) -> List[DetectionResult]:
        """Postprocess SSD model outputs."""
        # SSD models have 4 outputs: boxes, classes, scores, count
        # The order may vary, so we detect it on first run
        if not self._ssd_indices_determined:
            self._determine_ssd_indices()

        boxes = self.interpreter.get_tensor(self.output_details[self.output_boxes_index]['index'])[0]
        class_ids = self.interpreter.get_tensor(self.output_details[self.output_class_ids_index]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[self.output_scores_index]['index'])[0]
        count = int(self.interpreter.get_tensor(self.output_details[self.output_count_index]['index'])[0])

        results = []

        for i in range(min(count, self.max_detections)):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            class_id = int(class_ids[i])
            label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"

            # SSD boxes are [y_min, x_min, y_max, x_max] normalized
            y_min, x_min, y_max, x_max = boxes[i]

            results.append(DetectionResult(
                label=label,
                confidence=score,
                bounding_box=BoundingBox(
                    x_min=float(max(0.0, min(1.0, x_min))),
                    y_min=float(max(0.0, min(1.0, y_min))),
                    x_max=float(max(0.0, min(1.0, x_max))),
                    y_max=float(max(0.0, min(1.0, y_max)))
                )
            ))

        return results

    def _determine_ssd_indices(self):
        """Determine the output tensor indices for SSD models."""
        # Typical SSD output shapes (TFLite_Detection_PostProcess):
        # [0] boxes: (1, N, 4) - bounding boxes
        # [1] classes: (1, N) - class IDs (float32 but contains integers)
        # [2] scores: (1, N) - confidence scores
        # [3] count: (1,) - number of detections

        shape_2d_indices = []

        for i, details in enumerate(self.output_details):
            shape = details['shape']
            if len(shape) == 3 and shape[2] == 4:
                self.output_boxes_index = i
            elif len(shape) == 2:
                # Collect 2D tensors - we'll assign them by order
                shape_2d_indices.append(i)
            elif len(shape) == 1:
                self.output_count_index = i

        # For standard SSD models, 2D outputs are [class_ids, scores] in order
        if len(shape_2d_indices) >= 2:
            self.output_class_ids_index = shape_2d_indices[0]
            self.output_scores_index = shape_2d_indices[1]
        elif len(shape_2d_indices) == 1:
            # Only one 2D output - assume it's scores
            self.output_scores_index = shape_2d_indices[0]

        # Fallback to standard order if detection failed
        if self.output_boxes_index is None or self.output_class_ids_index is None:
            self.output_boxes_index = 0
            self.output_class_ids_index = 1
            self.output_scores_index = 2
            self.output_count_index = 3

        self._ssd_indices_determined = True

        logger.debug(
            f"SSD output indices: boxes={self.output_boxes_index}, "
            f"class_ids={self.output_class_ids_index}, scores={self.output_scores_index}, "
            f"count={self.output_count_index}"
        )

    def _postprocess_yolo(
        self, confidence_threshold: float, orig_width: int, orig_height: int
    ) -> List[DetectionResult]:
        """Postprocess YOLO model outputs with DFL decoding."""
        # Get raw outputs
        scores_raw = self.interpreter.tensor(self.scores_tensor_index)()[0]
        boxes_raw = self.interpreter.tensor(self.boxes_tensor_index)()[0]

        # Find candidates above threshold
        max_scores = np.max(scores_raw, axis=1)
        candidates = np.where(max_scores >= self.min_score_quantized)[0]

        if len(candidates) == 0:
            return []

        # Dequantize scores and apply sigmoid
        scores_quant = scores_raw[candidates].astype(np.float32)
        scores_dequant = (scores_quant - self.scores_zero_point) * self.scores_scale
        scores_dequant = scores_dequant + self.logit_shift
        scores = 1.0 / (1.0 + np.exp(-scores_dequant))  # Sigmoid

        # Get class predictions
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences >= confidence_threshold
        if not np.any(mask):
            return []

        candidates = candidates[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Dequantize and decode boxes using DFL
        boxes_quant = boxes_raw[candidates].astype(np.float32)
        boxes_dequant = (boxes_quant - self.boxes_zero_point) * self.boxes_scale

        # DFL decoding: reshape to (N, 4, reg_max) and apply softmax + projection
        boxes_reshaped = boxes_dequant.reshape(-1, 4, self.reg_max)
        boxes_softmax = self._softmax(boxes_reshaped, axis=2)
        boxes_decoded = np.sum(boxes_softmax * self.project, axis=2)

        # Convert from ltrb (left, top, right, bottom) offsets to xyxy
        anchors = self.anchors[candidates]
        strides = self.anchor_strides[candidates]

        lt = boxes_decoded[:, :2]  # left, top
        rb = boxes_decoded[:, 2:]  # right, bottom

        x1y1 = (anchors - lt) * strides
        x2y2 = (anchors + rb) * strides

        boxes_xyxy = np.concatenate([x1y1, x2y2], axis=1)

        # Normalize to [0, 1]
        boxes_xyxy[:, [0, 2]] /= self.input_width
        boxes_xyxy[:, [1, 3]] /= self.input_height

        # Apply NMS
        boxes_for_nms = boxes_xyxy.copy()
        boxes_for_nms[:, [0, 2]] *= orig_width
        boxes_for_nms[:, [1, 3]] *= orig_height

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            confidences.tolist(),
            confidence_threshold,
            self.iou_threshold
        )

        if len(indices) == 0:
            return []

        # Handle different OpenCV versions (indices may be nested)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        results = []
        for idx in indices:
            x_min, y_min, x_max, y_max = boxes_xyxy[idx]

            results.append(DetectionResult(
                label=self.labels[class_ids[idx]] if class_ids[idx] < len(self.labels) else f"class_{class_ids[idx]}",
                confidence=float(confidences[idx]),
                bounding_box=BoundingBox(
                    x_min=float(max(0.0, min(1.0, x_min))),
                    y_min=float(max(0.0, min(1.0, y_min))),
                    x_max=float(max(0.0, min(1.0, x_max))),
                    y_max=float(max(0.0, min(1.0, y_max)))
                )
            ))

        return results

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along specified axis."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def draw_detections(self, image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
        """
        Draw bounding boxes and labels on the image.

        Args:
            image: PIL Image to draw on
            detections: List of DetectionResult objects

        Returns:
            PIL Image with bounding boxes and labels drawn
        """
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

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

        for detection in detections:
            box = detection.bounding_box.to_pixel_coords(image.width, image.height)
            x_min, y_min = box["x_min"], box["y_min"]
            x_max, y_max = box["x_max"], box["y_max"]

            color = colors[hash(detection.label) % len(colors)]

            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)

            label_text = f"{detection.label}: {detection.confidence:.2f}"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_background_coords = [
                (x_min, max(0, y_min - text_height - 4)),
                (x_min + text_width + 4, y_min)
            ]

            draw.rectangle(text_background_coords, fill=color)
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
            "backend": "edgetpu",
            "model_type": self.model_type,
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "input_shape": list(self.input_shape),
            "num_labels": len(self.labels),
            "default_confidence_threshold": self.confidence_threshold,
            "device": self.device or "auto",
            "requires_int8": self.requires_int8,
            "tflite_source": TFLITE_SOURCE,
        }
