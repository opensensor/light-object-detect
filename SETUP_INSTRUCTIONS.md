# Setup Instructions for Enhanced light-object-detect

## Current Status

✓ Code enhancements complete
✓ COCO labels file created
⚠ Dependencies need to be installed
⚠ ONNX model needs to be downloaded

## Step-by-Step Setup

### Step 1: Install Python Dependencies

```bash
cd /home/matteius/lightNVR/light-object-detect

# Install all dependencies using pipenv
pipenv install

# This will install:
# - fastapi, uvicorn (API server)
# - pillow, numpy (image processing)
# - pydantic (data validation)
# - onnxruntime (ONNX backend)
# - opencv-python (OpenCV backend)
# - shapely (zone detection)
# - scipy (numerical operations)
```

### Step 2: Download Detection Model

You have **three options**:

#### Option A: Download YOLOv8 ONNX (Recommended - Best Performance)

**Method 1: Using Ultralytics (Easiest)**

```bash
# Install ultralytics
pip install ultralytics

# Download and export YOLOv8n model
python3 << 'EOF'
from ultralytics import YOLO
import shutil
from pathlib import Path

# Download YOLOv8n
model = YOLO('yolov8n.pt')

# Export to ONNX
onnx_path = model.export(format='onnx', simplify=True)

# Move to models directory
Path('models').mkdir(exist_ok=True)
shutil.move(onnx_path, 'models/yolov8n.onnx')
print(f"✓ Model saved to models/yolov8n.onnx")
EOF
```

**Method 2: Manual Download from Hugging Face**

```bash
cd models

# Remove empty file
rm -f yolov8n.onnx

# Download using curl (more reliable than wget for this)
curl -L -o yolov8n.onnx \
  "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx"

# Verify download
ls -lh yolov8n.onnx
# Should show ~6MB file

cd ..
```

**Method 3: Download using Python**

```bash
python3 << 'EOF'
import urllib.request
from pathlib import Path

url = "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx"
output = Path("models/yolov8n.onnx")

print("Downloading YOLOv8n ONNX model...")
urllib.request.urlretrieve(url, output)
print(f"✓ Downloaded to {output}")
print(f"  Size: {output.stat().st_size / 1024 / 1024:.1f} MB")
EOF
```

#### Option B: Use OpenCV DNN with YOLOv4

```bash
cd models

# Download YOLOv4-tiny (smaller, faster)
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

cd ..

# Configure for OpenCV backend
cat > .env << EOF
BACKEND=opencv
OPENCV_MODEL_PATH=models/yolov4-tiny.weights
OPENCV_CONFIG_PATH=models/yolov4-tiny.cfg
OPENCV_LABELS_PATH=models/coco_labels.txt
OPENCV_CONFIDENCE_THRESHOLD=0.5
OPENCV_NMS_THRESHOLD=0.4
EOF
```

#### Option C: Use Existing TFLite Model (If Available)

If you already have a TFLite model:

```bash
# Configure for TFLite backend
cat > .env << EOF
BACKEND=tflite
TFLITE_MODEL_PATH=models/detect.tflite
TFLITE_LABELS_PATH=models/labelmap.txt
TFLITE_CONFIDENCE_THRESHOLD=0.5
EOF
```

### Step 3: Configure the Backend

If you downloaded the ONNX model (recommended):

```bash
cat > .env << EOF
BACKEND=onnx
ONNX_MODEL_PATH=models/yolov8n.onnx
ONNX_LABELS_PATH=models/coco_labels.txt
ONNX_CONFIDENCE_THRESHOLD=0.5
ONNX_IOU_THRESHOLD=0.45
ONNX_MODEL_TYPE=yolov8
EOF
```

### Step 4: Verify Setup

```bash
# Run the setup verification script
pipenv run python3 test_setup.py
```

You should see:
```
✓ Basic setup is complete
✓ ONNX backend is ready
```

### Step 5: Start the API Server

```bash
# Start the server
pipenv run uvicorn main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
pipenv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 6: Test the API

In a new terminal:

```bash
# Download a test image
wget http://images.cocodataset.org/val2017/000000397133.jpg -O test.jpg

# Test basic detection
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@test.jpg" | jq

# Test with person filter
curl -X POST "http://localhost:8000/api/v1/detect?filter_classes=person" \
  -F "file=@test.jpg" | jq

# Test tracking
curl -X POST "http://localhost:8000/api/v1/detect-and-track?stream_name=test" \
  -F "file=@test.jpg" | jq
```

### Step 7: Integrate with lightNVR

```bash
# Rebuild lightNVR with updated detection structures
cd /home/matteius/lightNVR
make clean
make

# Update your config to use API detection
# Edit your config file to set:
#   "detection_backend": "api-detection"
#   "api_detection_url": "http://localhost:8000/api/v1/detect"

# Run lightNVR
./lightnvr
```

## Quick Commands Summary

```bash
# Complete setup in one go (using ultralytics for model download)
cd /home/matteius/lightNVR/light-object-detect

# 1. Install dependencies
pipenv install

# 2. Install ultralytics and download model
pip install ultralytics
python3 -c "from ultralytics import YOLO; import shutil; from pathlib import Path; Path('models').mkdir(exist_ok=True); model = YOLO('yolov8n.pt'); onnx_path = model.export(format='onnx', simplify=True); shutil.move(onnx_path, 'models/yolov8n.onnx')"

# 3. Configure
cat > .env << EOF
BACKEND=onnx
ONNX_MODEL_PATH=models/yolov8n.onnx
ONNX_LABELS_PATH=models/coco_labels.txt
ONNX_CONFIDENCE_THRESHOLD=0.5
ONNX_IOU_THRESHOLD=0.45
ONNX_MODEL_TYPE=yolov8
EOF

# 4. Start server
pipenv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### "pipenv: command not found"

```bash
pip install pipenv
# or
pip3 install pipenv
```

### "Model download fails"

Try the manual download method or use a different backend (OpenCV or TFLite).

### "CUDA not available" warning

This is normal if you don't have an NVIDIA GPU. The backends will use CPU, which is still fast enough for most use cases.

To enable GPU acceleration:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### "Port 8000 already in use"

```bash
# Use a different port
pipenv run uvicorn main:app --host 0.0.0.0 --port 8001

# Or find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

## What's New

All these features are now available:

✓ **Multiple Detection Backends**: ONNX Runtime, OpenCV DNN, TFLite
✓ **Zone-Based Detection**: Define polygon zones for targeted detection
✓ **Object Tracking**: Track objects across frames with unique IDs
✓ **Event Filtering**: Filter by class, confidence, size
✓ **lightNVR Integration**: Full support for tracking and zones in database

See [ENHANCEMENTS.md](ENHANCEMENTS.md) for detailed feature documentation.
See [QUICKSTART.md](QUICKSTART.md) for usage examples.

