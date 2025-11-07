# Quick Start Guide

This guide will help you get started with the enhanced light-object-detect API.

## Installation

### 1. Install Dependencies

```bash
cd light-object-detect
pipenv install
```

### 2. Download Models

You have several options for getting detection models:

#### Option A: Use TFLite Backend (Easiest - Lightweight)

The TFLite backend uses TensorFlow Lite for efficient inference on CPU. It's lightweight and works well on embedded systems.

**Install dependencies:**
```bash
pipenv install  # Installs TensorFlow which includes TFLite
```

**Download the model:**
```bash
# Download SSD MobileNet V1 TFLite model
bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v1

# Or for a more accurate model:
# bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v2
# bash scripts/download_tflite_models.sh --model-type efficientdet_lite0
```

**Configure:**
```bash
cat > .env << EOF
BACKEND=tflite
TFLITE_MODEL_PATH=backends/tflite/models/ssd_mobilenet_v1.tflite
TFLITE_LABELS_PATH=backends/tflite/models/coco_labels.txt
TFLITE_CONFIDENCE_THRESHOLD=0.5
EOF
```

**Note:** We use TensorFlow (which includes TFLite) instead of the deprecated `tflite-runtime` package. The backend automatically uses TensorFlow's TFLite interpreter. You may see a deprecation warning about migrating to `ai-edge-litert`, but this can be ignored for now as the new package doesn't support Python 3.12+.

#### Option B: Download YOLOv8 ONNX Model (Recommended for Best Performance)

**Method 1: Using Python Script (Requires ultralytics package)**

```bash
# Install ultralytics
pip install ultralytics

# Download and export YOLOv8n model
python3 scripts/download_models.py --model-size n --output-dir models

# Or for a larger, more accurate model:
python3 scripts/download_models.py --model-size s --output-dir models
```

**Method 2: Manual Download from Hugging Face**

1. Visit: https://huggingface.co/Ultralytics/YOLOv8/tree/main
2. Download `yolov8n.onnx` (or yolov8s.onnx for better accuracy)
3. Save to `light-object-detect/models/yolov8n.onnx`
4. Create labels file:

```bash
mkdir -p models
bash scripts/download_models.sh --output-dir models
```

**Method 3: Export from PyTorch**

```bash
pip install ultralytics
python3 << EOF
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', simplify=True)
EOF
mv yolov8n.onnx models/
```

Then configure:

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

#### Option C: Use OpenCV DNN Backend

Download a Darknet YOLO model:

```bash
mkdir -p models
cd models

# Download YOLOv4-tiny (smaller, faster)
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

# Or YOLOv4 (larger, more accurate)
# wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
# wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg

cd ..
```

Configure:

```bash
cat > .env << EOF
BACKEND=opencv
OPENCV_MODEL_PATH=models/yolov4-tiny.weights
OPENCV_CONFIG_PATH=models/yolov4-tiny.cfg
OPENCV_LABELS_PATH=models/coco_labels.txt
OPENCV_CONFIDENCE_THRESHOLD=0.5
OPENCV_NMS_THRESHOLD=0.4
EOF
```

### 3. Start the API Server

```bash
pipenv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Basic Usage

### Simple Detection

```bash
# Detect objects in an image
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@test_image.jpg"
```

### Detection with Specific Backend

```bash
# Use ONNX backend
curl -X POST "http://localhost:8000/api/v1/detect?backend=onnx" \
  -F "file=@test_image.jpg"

# Use OpenCV backend
curl -X POST "http://localhost:8000/api/v1/detect?backend=opencv" \
  -F "file=@test_image.jpg"
```

### Detection with Class Filtering

```bash
# Only detect persons
curl -X POST "http://localhost:8000/api/v1/detect?filter_classes=person" \
  -F "file=@test_image.jpg"

# Detect persons and cars
curl -X POST "http://localhost:8000/api/v1/detect?filter_classes=person,car" \
  -F "file=@test_image.jpg"
```

### Detection with Zones

```bash
# Define a zone (left half of image)
ZONES='{"zones":[{"id":"entrance","name":"Entrance","polygon":[{"x":0.0,"y":0.0},{"x":0.5,"y":0.0},{"x":0.5,"y":1.0},{"x":0.0,"y":1.0}],"enabled":true,"filter_classes":["person"]}],"zone_mode":"center"}'

curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@test_image.jpg" \
  --data-urlencode "zones=$ZONES"
```

### Detection with Tracking

```bash
# First frame
curl -X POST "http://localhost:8000/api/v1/detect-and-track?stream_name=camera1" \
  -F "file=@frame1.jpg"

# Second frame (objects will be tracked)
curl -X POST "http://localhost:8000/api/v1/detect-and-track?stream_name=camera1" \
  -F "file=@frame2.jpg"

# Get active tracks
curl "http://localhost:8000/api/v1/tracks/camera1"
```

## Testing with Sample Images

### Download Test Images

```bash
mkdir -p test_images
cd test_images

# Download sample images from COCO dataset
wget http://images.cocodataset.org/val2017/000000039769.jpg -O cats.jpg
wget http://images.cocodataset.org/val2017/000000397133.jpg -O people.jpg
wget http://images.cocodataset.org/val2017/000000037777.jpg -O traffic.jpg

cd ..
```

### Run Tests

```bash
# Test basic detection
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@test_images/people.jpg" | jq

# Test with person filter
curl -X POST "http://localhost:8000/api/v1/detect?filter_classes=person" \
  -F "file=@test_images/people.jpg" | jq

# Test tracking
curl -X POST "http://localhost:8000/api/v1/detect-and-track?stream_name=test" \
  -F "file=@test_images/people.jpg" | jq
```

## Integration with lightNVR

### 1. Configure lightNVR

Edit your lightNVR config to use API detection:

```json
{
  "streams": [
    {
      "name": "camera1",
      "url": "rtsp://camera-ip/stream",
      "detection_backend": "api-detection",
      "api_detection_url": "http://localhost:8000/api/v1/detect"
    }
  ],
  "api_detection_url": "http://localhost:8000/api/v1/detect"
}
```

### 2. Rebuild lightNVR

```bash
cd /home/matteius/lightNVR
make clean
make
```

### 3. Run lightNVR

```bash
./lightnvr
```

### 4. Query Detections from Database

```bash
# Get recent detections
sqlite3 /var/lib/lightnvr/lightnvr.db \
  "SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10;"

# Get detections with tracking info
sqlite3 /var/lib/lightnvr/lightnvr.db \
  "SELECT stream_name, label, confidence, track_id, zone_id, timestamp 
   FROM detections 
   WHERE track_id != -1 
   ORDER BY timestamp DESC LIMIT 20;"

# Get person detections in specific zone
sqlite3 /var/lib/lightnvr/lightnvr.db \
  "SELECT * FROM detections 
   WHERE label='person' AND zone_id='entrance' 
   ORDER BY timestamp DESC;"
```

## Performance Tips

### Backend Comparison

- **TFLite**: Lightweight, good for embedded systems, moderate accuracy
- **ONNX Runtime**: Best performance with CUDA, excellent accuracy with YOLOv8
- **OpenCV DNN**: Good balance, supports many formats, decent performance

### Model Size Comparison (YOLOv8)

- **yolov8n** (nano): ~6MB, fastest, good for real-time on CPU
- **yolov8s** (small): ~22MB, balanced speed/accuracy
- **yolov8m** (medium): ~52MB, better accuracy, slower
- **yolov8l** (large): ~87MB, high accuracy, requires GPU
- **yolov8x** (xlarge): ~136MB, best accuracy, requires powerful GPU

### Optimization

1. **Use CUDA**: Install `onnxruntime-gpu` for GPU acceleration
2. **Adjust confidence threshold**: Higher = fewer false positives
3. **Use zones**: Reduce processing by focusing on specific areas
4. **Enable tracking**: Reduces false positives with temporal consistency

## Troubleshooting

### Model Download Issues

If automatic download fails, manually download from:
- Hugging Face: https://huggingface.co/Ultralytics/YOLOv8/tree/main
- GitHub: https://github.com/ultralytics/assets/releases

### CUDA Not Available

```bash
# Check CUDA availability
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Install GPU version
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### API Connection Issues

```bash
# Check if API is running
curl http://localhost:8000/health

# Check logs
pipenv run uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

## Next Steps

- Read [ENHANCEMENTS.md](ENHANCEMENTS.md) for detailed feature documentation
- Explore the API docs at http://localhost:8000/docs
- Configure zones for your specific use case
- Set up tracking for your video streams
- Integrate with lightNVR for complete NVR solution

