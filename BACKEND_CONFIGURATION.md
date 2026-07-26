# Detection Backend Configuration Guide

This guide explains how to configure different detection backends for both light-object-detect API and lightNVR.

## Overview

The light-object-detect API supports three detection backends:
- **ONNX Runtime**: Best performance with CUDA, excellent accuracy with YOLOv8
- **TFLite**: Lightweight, good for embedded systems, moderate accuracy
- **OpenCV DNN**: Good balance, supports many formats, decent performance

## 1. Download Models

### TFLite Models

Download TFLite models using the provided script:

```bash
cd light-object-detect

# Download SSD MobileNet V1 (recommended for embedded systems)
bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v1

# Or download SSD MobileNet V2 (more accurate)
bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v2

# Or download EfficientDet Lite0 (best accuracy for TFLite)
bash scripts/download_tflite_models.sh --model-type efficientdet_lite0
```

This will download the model to `backends/tflite/models/` and create the labels file.

### ONNX Models

Download YOLOv8 ONNX models:

```bash
# Using the download script (requires wget)
bash scripts/download_models.sh --model-size n --output-dir models

# Or using Python (requires ultralytics package)
pip install ultralytics
python3 scripts/download_models.py --model-size n --output-dir models
```

Available model sizes:
- `n` (nano): ~6MB, fastest
- `s` (small): ~22MB, balanced
- `m` (medium): ~52MB, better accuracy
- `l` (large): ~87MB, high accuracy
- `x` (xlarge): ~136MB, best accuracy

### OpenCV DNN Models

Download YOLO models for OpenCV:

```bash
cd models

# YOLOv4-tiny (smaller, faster)
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

# Or YOLOv4 (larger, more accurate)
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
```

## 2. Configure light-object-detect API

The API can be configured via environment variables or query parameters.

### Environment Variables (.env file)

**For TFLite:**
```bash
BACKEND=tflite
TFLITE_MODEL_PATH=backends/tflite/models/ssd_mobilenet_v1.tflite
TFLITE_LABELS_PATH=backends/tflite/models/coco_labels.txt
TFLITE_CONFIDENCE_THRESHOLD=0.5
```

**For ONNX:**
```bash
BACKEND=onnx
ONNX_MODEL_PATH=models/yolov8n.onnx
ONNX_LABELS_PATH=models/coco_labels.txt
ONNX_CONFIDENCE_THRESHOLD=0.5
ONNX_IOU_THRESHOLD=0.45
ONNX_MODEL_TYPE=yolov8
```

**For OpenCV:**
```bash
BACKEND=opencv
OPENCV_MODEL_PATH=models/yolov4-tiny.weights
OPENCV_CONFIG_PATH=models/yolov4-tiny.cfg
OPENCV_LABELS_PATH=models/coco_labels.txt
OPENCV_CONFIDENCE_THRESHOLD=0.5
OPENCV_NMS_THRESHOLD=0.4
```

### Query Parameters

You can override the backend per request:

```bash
# Use TFLite backend
curl -X POST 'http://localhost:9001/api/v1/detect?backend=tflite' \
  -F 'file=@image.jpg'

# Use ONNX backend
curl -X POST 'http://localhost:9001/api/v1/detect?backend=onnx' \
  -F 'file=@image.jpg'

# Use OpenCV backend
curl -X POST 'http://localhost:9001/api/v1/detect?backend=opencv' \
  -F 'file=@image.jpg'
```

## 3. Configure lightNVR

lightNVR can be configured to use a specific backend for all API detection calls.

### Edit lightnvr.ini

```ini
[api_detection]
url = http://localhost:9001/detect
backend = tflite  ; Options: onnx, tflite, opencv (default: onnx)
```

### Per-Stream Custom Endpoint

You can also override the detection endpoint per stream using the web UI:

1. Go to **Streams** page
2. Click **Edit** on a stream
3. Enable **AI Detection Recording**
4. Select **API Detection (light-object-detect)** from the dropdown
5. Click **Override with Custom Endpoint**
6. Enter your custom URL (e.g., `http://192.168.1.100:9001/detect`)
7. Save the stream

The custom URL will be stored in the database and used for that specific stream.

## 4. Start the Services

### Start light-object-detect API

```bash
cd light-object-detect
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001
```

### Start lightNVR

```bash
cd /home/matteius/lightNVR
./bin/lightnvr
```

## 5. Test Detection

### Test light-object-detect API

```bash
# Download a test image
wget http://images.cocodataset.org/val2017/000000397133.jpg -O test.jpg

# Test with TFLite backend
curl -X POST 'http://localhost:9001/api/v1/detect?backend=tflite' \
  -F 'file=@test.jpg' | jq

# Test with ONNX backend
curl -X POST 'http://localhost:9001/api/v1/detect?backend=onnx' \
  -F 'file=@test.jpg' | jq
```

### Check lightNVR Logs

```bash
# Check if lightNVR is calling the API
tail -f /var/lib/lightnvr/data/logs/lightnvr.log | grep "API Detection"
```

You should see log messages like:
```
API Detection: Using URL with parameters: http://localhost:9001/detect?backend=tflite&confidence_threshold=0.5&return_image=false (backend: tflite)
```

## 6. Performance Comparison

| Backend | Speed | Accuracy | Memory | Best For |
|---------|-------|----------|--------|----------|
| **TFLite** | Fast | Good | Low | Embedded systems, Raspberry Pi |
| **ONNX** | Very Fast* | Excellent | Medium | Desktop, servers with GPU |
| **OpenCV** | Medium | Good | Medium | General purpose, CPU-only |

*With CUDA GPU acceleration

## 7. Troubleshooting

### TFLite Model Not Found

```bash
# Download the model
cd light-object-detect
bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v1

# Verify the file exists
ls -lh backends/tflite/models/ssd_mobilenet_v1.tflite
```

### ONNX Model Not Found

```bash
# Download the model
cd light-object-detect
bash scripts/download_models.sh --model-size n --output-dir models

# Verify the file exists
ls -lh models/yolov8n.onnx
```

### lightNVR Not Calling API

1. Check if light-object-detect is running:
   ```bash
   curl http://localhost:9001/health
   ```

2. Check lightNVR config:
   ```bash
   grep -A 2 "\[api_detection\]" config/lightnvr.ini
   ```

3. Check lightNVR logs:
   ```bash
   tail -f /var/lib/lightnvr/data/logs/lightnvr.log | grep "API Detection"
   ```

### Wrong Backend Being Used

Check the lightNVR logs to see which backend is being used:
```bash
tail -f /var/lib/lightnvr/data/logs/lightnvr.log | grep "backend:"
```

The log should show:
```
API Detection: Using URL with parameters: ... (backend: tflite)
```

If it's using the wrong backend, update `config/lightnvr.ini`:
```ini
[api_detection]
backend = tflite
```

Then restart lightNVR.

## 8. Advanced Configuration

### GPU Acceleration for ONNX

Install ONNX Runtime with CUDA support:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

Check if CUDA is available:
```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### OpenVINO Acceleration for ONNX (Intel CPU / iGPU / NPU)

OpenVINO runs the same ONNX model on Intel hardware — including the integrated
GPU and the NPU found on Core Ultra parts — and is typically several times
faster than the plain CPU provider on the same machine.

Install ONNX Runtime with OpenVINO support. This is a **drop-in replacement**
for `onnxruntime`, not an addition — having both installed breaks the import:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-openvino
```

Confirm the provider registered:
```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect 'OpenVINOExecutionProvider' in the list
```

Configure it in `.env`:

```bash
# Preference order. Providers missing from the installed wheel are skipped,
# so this same value is safe on a CUDA box and on a plain-CPU box.
ONNX_EXECUTION_PROVIDERS=openvino,cpu

# AUTO lets OpenVINO pick the best device it can see. Pin it with CPU, GPU or
# NPU, or spread work with a device list such as MULTI:GPU,CPU.
ONNX_OPENVINO_DEVICE_TYPE=AUTO

# Optional tuning
#ONNX_OPENVINO_PRECISION=FP16          # FP32, FP16 or ACCURACY
#ONNX_OPENVINO_NUM_THREADS=4           # CPU device only
#ONNX_OPENVINO_CACHE_DIR=/tmp/ov_cache # caches compiled blobs
```

Set `ONNX_OPENVINO_CACHE_DIR` if you target GPU or NPU. The first session
compiles the model for the device, which can take tens of seconds; the cache
turns every subsequent start into a load. Point it at a persistent volume in
Docker or the cache is rebuilt on every container start.

**Device access.** The CPU device needs nothing extra. The iGPU additionally
needs `intel-opencl-icd` on the host and `--device /dev/dri` passed into the
container; the NPU needs the `intel-driver-compiler-npu` / `intel-level-zero-npu`
packages and `--device /dev/accel`.

Startup logs which providers the session actually got:
```
ONNX: session for backends/onnx/models/yolo11n.onnx using providers ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
```

If OpenVINO cannot compile the model for the requested device — most common when
pinning `NPU` — the backend logs a warning and falls back to CPU rather than
failing the whole API. Watch for that warning if throughput looks unchanged
after switching.

### Custom Confidence Thresholds

You can set different confidence thresholds per backend in the `.env` file:

```bash
TFLITE_CONFIDENCE_THRESHOLD=0.6
ONNX_CONFIDENCE_THRESHOLD=0.5
OPENCV_CONFIDENCE_THRESHOLD=0.55
```

Or override via query parameter:
```bash
curl -X POST 'http://localhost:9001/api/v1/detect?backend=tflite&confidence_threshold=0.7' \
  -F 'file=@image.jpg'
```

## 9. Summary

- **TFLite**: Best for embedded systems (Raspberry Pi, low-power devices)
- **ONNX**: Best for servers with GPU acceleration, and for Intel hardware via OpenVINO
- **OpenCV**: Good general-purpose option for CPU-only systems

Choose the backend that best fits your hardware and accuracy requirements!

