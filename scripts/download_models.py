#!/usr/bin/env python3
"""
Script to download and export YOLOv8 models to ONNX format.
"""

import os
import sys
import argparse
from pathlib import Path

def download_yolov8_onnx(model_size='n', output_dir='backends/onnx/models'):
    """
    Download YOLOv8 model and export to ONNX format.

    Args:
        model_size: Model size (n, s, m, l, x)
        output_dir: Directory to save the model
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics package not found.")
        print("Install it with: pip install ultralytics")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name = f'yolov8{model_size}.pt'
    onnx_name = f'yolov8{model_size}.onnx'
    
    print(f"Downloading YOLOv8{model_size} model...")
    model = YOLO(model_name)

    print(f"Exporting to ONNX format (opset 21 for compatibility)...")
    onnx_path = model.export(format='onnx', simplify=True, opset=21)
    
    # Move to output directory
    final_path = output_path / onnx_name
    if os.path.exists(onnx_path):
        os.rename(onnx_path, final_path)
        print(f"✓ Model saved to: {final_path}")
    else:
        print(f"ERROR: Export failed, file not found: {onnx_path}")
        sys.exit(1)
    
    return final_path


def create_coco_labels(output_dir='backends/onnx/models'):
    """Create COCO labels file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    labels_file = output_path / 'coco_labels.txt'
    
    # COCO 80 classes
    coco_labels = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    with open(labels_file, 'w') as f:
        for label in coco_labels:
            f.write(f"{label}\n")
    
    print(f"✓ Labels saved to: {labels_file}")
    return labels_file


def main():
    parser = argparse.ArgumentParser(description='Download YOLOv8 models and create label files')
    parser.add_argument(
        '--model-size',
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--output-dir',
        default='backends/onnx/models',
        help='Output directory for models and labels'
    )
    parser.add_argument(
        '--labels-only',
        action='store_true',
        help='Only create labels file, skip model download'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8 Model Download and Export Tool")
    print("=" * 60)
    
    # Create labels file
    labels_file = create_coco_labels(args.output_dir)
    
    if not args.labels_only:
        # Download and export model
        model_file = download_yolov8_onnx(args.model_size, args.output_dir)
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print(f"\nModel: {model_file}")
        print(f"Labels: {labels_file}")
        print("\nUpdate your .env file with:")
        print(f"BACKEND=onnx")
        print(f"ONNX_MODEL_PATH={model_file}")
        print(f"ONNX_LABELS_PATH={labels_file}")
        print(f"ONNX_MODEL_TYPE=yolov8")
    else:
        print("\n" + "=" * 60)
        print("Labels file created!")
        print("=" * 60)


if __name__ == '__main__':
    main()

