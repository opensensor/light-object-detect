import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont

from backends.base import DetectionBackend
from models.detection import DetectionResult, BoundingBox

logger = logging.getLogger(__name__)


class MoondreamBackend(DetectionBackend):
    """Moondream VLM backend using HuggingFace Transformers for local inference.

    Loads the Moondream2 model weights directly in-process — no external
    API key or server required.  Works on CPU and GPU (uses bfloat16
    to match the native weight dtype of the model).

    Requires ``transformers>=4.51.1,<5.0``.
    """

    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        revision: Optional[str] = None,
        device: str = "cpu",
        default_detect_classes: Optional[List[str]] = None,
    ):
        self._model_name = model_name
        self._device = device

        # The Moondream2 weights are stored in bfloat16.  Modern CPUs
        # handle bfloat16 fine (PyTorch emulates it if needed), and
        # loading in float32 can cause numerical instability during
        # inference, so we default to bfloat16 everywhere.
        dtype = torch.bfloat16

        logger.info(
            f"Loading Moondream model '{model_name}' on {device} "
            f"(dtype={dtype}) …"
        )

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "device_map": device,
        }
        if revision:
            load_kwargs["revision"] = revision

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )

        self.model.eval()
        self.default_detect_classes = default_detect_classes or ["person", "car"]
        logger.info("MoondreamBackend ready (local transformers)")

    # ------------------------------------------------------------------
    # DetectionBackend abstract methods
    # ------------------------------------------------------------------

    def detect(
        self, image: Image.Image, confidence_threshold: float = 0.5,
        object_classes: Optional[List[str]] = None,
    ) -> List[DetectionResult]:
        """Detect objects by running Moondream detect for each class."""
        classes = object_classes or self.default_detect_classes
        results: List[DetectionResult] = []

        for cls in classes:
            try:
                output = self.model.detect(image, cls)
                for region in output.get("objects", []):
                    x_min = region.get("x_min", 0.0)
                    y_min = region.get("y_min", 0.0)
                    x_max = region.get("x_max", 0.0)
                    y_max = region.get("y_max", 0.0)
                    results.append(
                        DetectionResult(
                            label=cls,
                            confidence=1.0,
                            bounding_box=BoundingBox(
                                x_min=x_min,
                                y_min=y_min,
                                x_max=x_max,
                                y_max=y_max,
                            ),
                        )
                    )
            except Exception as e:
                logger.warning(f"Moondream detect failed for class '{cls}': {e}")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "backend": "moondream",
            "model": self._model_name,
            "device": self._device,
            "capabilities": ["detect", "describe", "query"],
            "default_detect_classes": self.default_detect_classes,
        }

    def draw_detections(
        self, image: Image.Image, detections: List[DetectionResult],
    ) -> Image.Image:
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255),
        ]

        for det in detections:
            box = det.bounding_box.to_pixel_coords(image.width, image.height)
            x_min, y_min = box["x_min"], box["y_min"]
            x_max, y_max = box["x_max"], box["y_max"]
            color = colors[hash(det.label) % len(colors)]
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)
            label_text = f"{det.label}: {det.confidence:.2f}"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            draw.rectangle(
                [(x_min, max(0, y_min - th - 4)), (x_min + tw + 4, y_min)],
                fill=color,
            )
            draw.text(
                (x_min + 2, max(0, y_min - th - 2)),
                label_text, fill=(255, 255, 255), font=font,
            )

        return draw_image

    # ------------------------------------------------------------------
    # VLM-specific capabilities
    # ------------------------------------------------------------------

    def describe(self, image: Image.Image, length: str = "normal") -> str:
        output = self.model.caption(image, length=length)
        return output.get("caption", "")

    def query(self, image: Image.Image, question: str) -> str:
        output = self.model.query(image, question)
        return output.get("answer", "")

