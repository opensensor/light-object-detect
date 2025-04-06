#!/usr/bin/env python3
"""
Script to download a sample TFLite model for object detection.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def download_file(url, destination):
    """Download a file from a URL to a destination."""
    print(f"Downloading {url} to {destination}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download the file
    urllib.request.urlretrieve(url, destination)
    
    print(f"Downloaded {destination}")


def extract_zip(zip_path, extract_dir):
    """Extract a zip file to a directory."""
    print(f"Extracting {zip_path} to {extract_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"Extracted {zip_path}")


def download_ssd_mobilenet():
    """Download SSD MobileNet v1 model."""
    model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
    zip_path = "/tmp/ssd_mobilenet.zip"
    extract_dir = "/tmp/ssd_mobilenet"
    model_dir = os.path.dirname(settings.TFLITE_MODEL_PATH)
    
    # Download the model
    download_file(model_url, zip_path)
    
    # Extract the zip file
    extract_zip(zip_path, extract_dir)
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy the model file
    shutil.copy(
        os.path.join(extract_dir, "detect.tflite"),
        settings.TFLITE_MODEL_PATH
    )
    
    # Create a labelmap file
    with open(settings.TFLITE_LABELS_PATH, 'w') as f:
        # COCO dataset labels
        labels = [
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
        for label in labels:
            f.write(f"{label}\n")
    
    # Clean up
    os.remove(zip_path)
    shutil.rmtree(extract_dir)
    
    print(f"Model saved to {settings.TFLITE_MODEL_PATH}")
    print(f"Labels saved to {settings.TFLITE_LABELS_PATH}")


def main():
    """Main function."""
    print("Downloading TFLite model for object detection...")
    
    # Download SSD MobileNet model
    download_ssd_mobilenet()
    
    print("Done!")


if __name__ == "__main__":
    main()
