from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence
import tempfile
import urllib.request
import zipfile


TFLITE_SSD_MOBILENET_V1_ZIP_URL: Final[str] = (
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
)


COCO_LABELS: Final[Sequence[str]] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


class ModelDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnsureModelResult:
    ok: bool
    did_download: bool
    message: str


def _write_labels_file(labels_path: Path) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("".join(f"{label}\n" for label in COCO_LABELS), encoding="utf-8")


def ensure_tflite_ssd_mobilenet_v1(
    model_path: str,
    labels_path: str,
    *,
    force: bool = False,
    timeout_seconds: float = 60.0,
) -> EnsureModelResult:
    """
    Ensure the default SSD MobileNet v1 TFLite model + labels exist locally.

    - If files are present, nothing is downloaded.
    - If missing (or force=True), the model zip is downloaded and `detect.tflite` is extracted.
    """
    model_file = Path(model_path)
    labels_file = Path(labels_path)

    if not force and model_file.exists():
        if not labels_file.exists():
            _write_labels_file(labels_file)
            return EnsureModelResult(
                ok=True,
                did_download=False,
                message=f"Model already exists. Labels were created at '{labels_file.as_posix()}'.",
            )
        return EnsureModelResult(ok=True, did_download=False, message="Model and labels already exist.")

    model_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="light-object-detect-") as tmpdir:
            tmp_dir = Path(tmpdir)
            zip_dest = tmp_dir / "model.zip"

            # Download
            req = urllib.request.Request(
                TFLITE_SSD_MOBILENET_V1_ZIP_URL,
                headers={"User-Agent": "light-object-detect/0.1"},
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                zip_dest.write_bytes(resp.read())

            # Extract expected file
            with zipfile.ZipFile(zip_dest) as zf:
                member_name = "detect.tflite"
                try:
                    extracted = zf.extract(member_name, path=tmp_dir)
                except KeyError as e:
                    raise ModelDownloadError(
                        f"Zip did not contain expected '{member_name}'."
                    ) from e

            extracted_path = Path(extracted)
            if not extracted_path.exists():
                raise ModelDownloadError("Extraction succeeded but file is missing on disk.")

            # Atomic-ish replace
            tmp_model = tmp_dir / "detect.tmp.tflite"
            tmp_model.write_bytes(extracted_path.read_bytes())
            tmp_model.replace(model_file)

            if not labels_file.exists() or force:
                _write_labels_file(labels_file)

        return EnsureModelResult(
            ok=True,
            did_download=True,
            message=f"Downloaded model to '{model_file.as_posix()}' and ensured labels at '{labels_file.as_posix()}'.",
        )
    except ModelDownloadError:
        raise
    except Exception as e:
        raise ModelDownloadError(f"Failed to ensure TFLite model: {e}") from e

