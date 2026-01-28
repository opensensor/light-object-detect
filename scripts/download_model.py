#!/usr/bin/env python3
"""
Script to download a default TFLite model for object detection.

This is mainly intended for local development; Docker builds can also run it
to bake the model into the image.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path to import config + utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from utils.model_download import ensure_tflite_ssd_mobilenet_v1, ModelDownloadError


def main():
    """Main function."""
    print("Ensuring default TFLite model for object detection...")

    try:
        result = ensure_tflite_ssd_mobilenet_v1(
            model_path=settings.TFLITE_MODEL_PATH,
            labels_path=settings.TFLITE_LABELS_PATH,
            force=False,
        )
    except ModelDownloadError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1) from e

    print(result.message)
    print("Done.")


if __name__ == "__main__":
    main()
