#!/usr/bin/env python3
"""
Test script to verify the setup and download models if needed.
"""

import os
import sys
from pathlib import Path

def check_labels():
    """Check if COCO labels file exists."""
    labels_path = Path("models/coco_labels.txt")
    if labels_path.exists():
        with open(labels_path) as f:
            labels = f.read().strip().split('\n')
        print(f"✓ Labels file found: {len(labels)} classes")
        return True
    else:
        print("✗ Labels file not found")
        return False

def check_onnx_model():
    """Check if ONNX model exists and is valid."""
    model_path = Path("models/yolov8n.onnx")
    if model_path.exists():
        size = model_path.stat().st_size
        if size > 1000000:  # At least 1MB
            print(f"✓ ONNX model found: {size / 1024 / 1024:.1f} MB")
            return True
        else:
            print(f"✗ ONNX model file is too small ({size} bytes), likely invalid")
            return False
    else:
        print("✗ ONNX model not found")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    packages = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pydantic': 'Pydantic',
    }
    
    optional_packages = {
        'onnxruntime': 'ONNX Runtime',
        'cv2': 'OpenCV',
        'shapely': 'Shapely',
    }
    
    print("\nChecking required dependencies:")
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            all_ok = False
    
    print("\nChecking optional dependencies:")
    for module, name in optional_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - Not installed (optional)")
    
    return all_ok

def download_onnx_model():
    """Try to download ONNX model using ultralytics."""
    print("\nAttempting to download YOLOv8n ONNX model...")
    try:
        from ultralytics import YOLO
        print("  ✓ Ultralytics package found")
        
        print("  Downloading YOLOv8n...")
        model = YOLO('yolov8n.pt')
        
        print("  Exporting to ONNX...")
        onnx_path = model.export(format='onnx', simplify=True)
        
        # Move to models directory
        import shutil
        dest = Path("models/yolov8n.onnx")
        shutil.move(onnx_path, dest)
        
        print(f"  ✓ Model saved to {dest}")
        return True
        
    except ImportError:
        print("  ✗ Ultralytics package not installed")
        print("\n  Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("light-object-detect Setup Verification")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    print("Checking Models and Data")
    print("=" * 60)
    
    # Check labels
    labels_ok = check_labels()
    
    # Check ONNX model
    onnx_ok = check_onnx_model()
    
    if not onnx_ok:
        print("\nONNX model not found or invalid.")
        response = input("Would you like to try downloading it now? (y/n): ")
        if response.lower() == 'y':
            onnx_ok = download_onnx_model()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if deps_ok and labels_ok:
        print("✓ Basic setup is complete")
        
        if onnx_ok:
            print("✓ ONNX backend is ready")
            print("\nYou can start the server with:")
            print("  BACKEND=onnx pipenv run uvicorn main:app --host 0.0.0.0 --port 8000")
        else:
            print("⚠ ONNX model not available")
            print("\nOptions:")
            print("1. Install ultralytics and run this script again:")
            print("   pip install ultralytics")
            print("   python3 test_setup.py")
            print("\n2. Download manually from:")
            print("   https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx")
            print("   Save to: models/yolov8n.onnx")
            print("\n3. Use TFLite backend if you have a .tflite model")
    else:
        print("✗ Setup incomplete")
        if not deps_ok:
            print("\nInstall dependencies with:")
            print("  pipenv install")
        if not labels_ok:
            print("\nCreate labels file with:")
            print("  bash scripts/download_models.sh")

if __name__ == '__main__':
    main()

