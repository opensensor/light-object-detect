#!/usr/bin/env python3
"""
Download pre-compiled EdgeTPU models from the Coral Model Zoo.

Usage:
    python download_models.py [--model MODEL_NAME] [--output-dir DIR]
    
Available models:
    - ssd_mobilenet_v2 (default) - SSD MobileNet V2 COCO
    - ssd_mobilenet_v1 - SSD MobileNet V1 COCO
    - efficientdet_lite0 - EfficientDet-Lite0 COCO
    - efficientdet_lite1 - EfficientDet-Lite1 COCO
    - efficientdet_lite2 - EfficientDet-Lite2 COCO
    - yolov5n - YOLOv5 Nano (requires custom compilation)
"""
import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path


# Coral Model Zoo URLs
CORAL_MODELS = {
    "ssd_mobilenet_v2": {
        "url": "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        "filename": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        "description": "SSD MobileNet V2 COCO - Fast, good accuracy",
    },
    "ssd_mobilenet_v1": {
        "url": "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
        "filename": "ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
        "description": "SSD MobileNet V1 COCO - Fastest, lower accuracy",
    },
    "efficientdet_lite0": {
        "url": "https://github.com/google-coral/test_data/raw/master/efficientdet_lite0_320_ptq_edgetpu.tflite",
        "filename": "efficientdet_lite0_320_ptq_edgetpu.tflite",
        "description": "EfficientDet-Lite0 - Balanced speed/accuracy",
    },
    "efficientdet_lite1": {
        "url": "https://github.com/google-coral/test_data/raw/master/efficientdet_lite1_384_ptq_edgetpu.tflite",
        "filename": "efficientdet_lite1_384_ptq_edgetpu.tflite",
        "description": "EfficientDet-Lite1 - Higher accuracy, slower",
    },
    "efficientdet_lite2": {
        "url": "https://github.com/google-coral/test_data/raw/master/efficientdet_lite2_448_ptq_edgetpu.tflite",
        "filename": "efficientdet_lite2_448_ptq_edgetpu.tflite",
        "description": "EfficientDet-Lite2 - Highest accuracy, slowest",
    },
}

# COCO labels URL
COCO_LABELS_URL = "https://github.com/google-coral/test_data/raw/master/coco_labels.txt"


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress indication."""
    print(f"Downloading {description or url}...")
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print(f"\n  Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_model(model_name: str, output_dir: Path) -> bool:
    """Download a specific model."""
    if model_name not in CORAL_MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(CORAL_MODELS.keys())}")
        return False
    
    model_info = CORAL_MODELS[model_name]
    output_path = output_dir / model_info["filename"]
    
    if output_path.exists():
        print(f"Model already exists: {output_path}")
        return True
    
    return download_file(
        model_info["url"],
        output_path,
        f"{model_name} ({model_info['description']})"
    )


def download_labels(output_dir: Path) -> bool:
    """Download COCO labels file."""
    output_path = output_dir / "coco_labels.txt"
    
    if output_path.exists():
        print(f"Labels already exist: {output_path}")
        return True
    
    return download_file(COCO_LABELS_URL, output_path, "COCO labels")


def main():
    parser = argparse.ArgumentParser(description="Download EdgeTPU models from Coral Model Zoo")
    parser.add_argument(
        "--model", "-m",
        default="ssd_mobilenet_v2",
        choices=list(CORAL_MODELS.keys()) + ["all"],
        help="Model to download (default: ssd_mobilenet_v2)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: backends/edgetpu/models)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available EdgeTPU models:")
        for name, info in CORAL_MODELS.items():
            print(f"  {name}: {info['description']}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "models"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Download labels first
    if not download_labels(output_dir):
        print("Warning: Failed to download labels")
    
    # Download model(s)
    if args.model == "all":
        for model_name in CORAL_MODELS:
            download_model(model_name, output_dir)
    else:
        download_model(args.model, output_dir)
    
    print("\nDone! To use the model, update your .env or config:")
    print(f'  EDGETPU_MODEL_PATH="{output_dir / CORAL_MODELS[args.model]["filename"]}"')
    print(f'  EDGETPU_LABELS_PATH="{output_dir / "coco_labels.txt"}"')


if __name__ == "__main__":
    main()

