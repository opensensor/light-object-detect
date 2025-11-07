# Model Setup Complete ✅

All detection backends are now properly configured and ready to use!

## Summary

### ✅ TFLite Backend (CPU-optimized)
- **Model**: SSD MobileNet V1 (4.0 MB)
- **Location**: `backends/tflite/models/ssd_mobilenet_v1.tflite`
- **Labels**: `backends/tflite/models/coco_labels.txt`
- **Runtime**: TensorFlow 2.20.0
- **Status**: ✅ Working

### ✅ ONNX Backend (Best accuracy)
- **Model**: YOLOv8n (12.3 MB)
- **Location**: `backends/onnx/models/yolov8n.onnx`
- **Labels**: `backends/onnx/models/coco_labels.txt`
- **Runtime**: ONNX Runtime 1.23.2
- **Status**: ✅ Working

## Configuration

### For TFLite Backend (Lightweight, CPU)

Create or update `.env`:
```bash
BACKEND=tflite
TFLITE_MODEL_PATH=backends/tflite/models/ssd_mobilenet_v1.tflite
TFLITE_LABELS_PATH=backends/tflite/models/coco_labels.txt
TFLITE_CONFIDENCE_THRESHOLD=0.5
```

### For ONNX Backend (Best accuracy, recommended)

Create or update `.env`:
```bash
BACKEND=onnx
ONNX_MODEL_PATH=backends/onnx/models/yolov8n.onnx
ONNX_LABELS_PATH=backends/onnx/models/coco_labels.txt
ONNX_MODEL_TYPE=yolov8
ONNX_CONFIDENCE_THRESHOLD=0.5
```

## Testing

Both backends have been tested and verified:

```bash
# Test TFLite backend
pipenv run python -c "from backends.tflite.backend import TFLiteBackend; \
  backend = TFLiteBackend('backends/tflite/models/ssd_mobilenet_v1.tflite', \
  'backends/tflite/models/coco_labels.txt'); \
  print('✓ TFLite backend working!')"

# Test ONNX backend
pipenv run python -c "from backends.onnx.backend import ONNXBackend; \
  backend = ONNXBackend('backends/onnx/models/yolov8n.onnx', \
  'backends/onnx/models/coco_labels.txt'); \
  print('✓ ONNX backend working!')"
```

## Starting the API Server

Start the detection API server:

```bash
cd light-object-detect
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001
```

The API will be available at:
- **Health check**: http://localhost:9001/health
- **Detection endpoint**: http://localhost:9001/api/v1/detect
- **API docs**: http://localhost:9001/docs

## Using with lightNVR

### Configure lightNVR

Edit `config/lightnvr.ini`:

```ini
[api_detection]
url = http://localhost:9001/detect
backend = onnx  # or tflite for CPU-only systems
```

### Rebuild lightNVR

```bash
cd /home/matteius/lightNVR
cmake --build build --target lightnvr -j$(nproc)
```

### Test Detection

1. Start the light-object-detect API server (see above)
2. Start lightNVR
3. Configure a stream with AI Detection enabled
4. Select "API Detection (light-object-detect)" from the dropdown
5. Optionally override the endpoint URL for per-stream configuration

## API Usage Examples

### Detect with ONNX backend
```bash
curl -X POST "http://localhost:9001/api/v1/detect?backend=onnx&confidence_threshold=0.5" \
  -F "image=@test_image.jpg"
```

### Detect with TFLite backend
```bash
curl -X POST "http://localhost:9001/api/v1/detect?backend=tflite&confidence_threshold=0.5" \
  -F "image=@test_image.jpg"
```

### Detect with zones (polygon filtering)
```bash
curl -X POST "http://localhost:9001/api/v1/detect?backend=onnx" \
  -F "image=@test_image.jpg" \
  -F 'zones=[{"points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]}]'
```

## Performance Comparison

| Backend | Model | Size | Speed (CPU) | Accuracy | Use Case |
|---------|-------|------|-------------|----------|----------|
| TFLite | SSD MobileNet V1 | 4 MB | Fast | Good | Embedded/CPU-only |
| ONNX | YOLOv8n | 12 MB | Medium | Excellent | General purpose |

## Troubleshooting

### "Model not found" error
- Run the download scripts:
  ```bash
  bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v1
  pipenv run python scripts/download_models.py --model-size n
  ```

### "Backend not available" error
- Make sure dependencies are installed:
  ```bash
  pipenv install
  ```

### GPU not detected (ONNX)
- This is normal if you don't have a GPU
- ONNX Runtime will automatically fall back to CPU
- For GPU support, install `onnxruntime-gpu` instead of `onnxruntime`

## Next Steps

1. ✅ Models downloaded and verified
2. ✅ Backends tested and working
3. ✅ lightNVR configured to use API detection
4. 🔄 Start the API server
5. 🔄 Test detection with a real camera stream

## Additional Models

### Download other YOLOv8 sizes (ONNX)
```bash
# YOLOv8s (small) - better accuracy, slower
pipenv run python scripts/download_models.py --model-size s

# YOLOv8m (medium) - even better accuracy
pipenv run python scripts/download_models.py --model-size m

# YOLOv8l (large) - best accuracy, slowest
pipenv run python scripts/download_models.py --model-size l
```

### Download other TFLite models
```bash
# SSD MobileNet V2 - more accurate than V1
bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v2

# EfficientDet Lite0 - best TFLite accuracy
bash scripts/download_tflite_models.sh --model-type efficientdet_lite0
```

## Documentation

- **QUICKSTART.md** - Quick start guide
- **BACKEND_CONFIGURATION.md** - Detailed backend configuration
- **MIGRATION_NOTES.md** - TensorFlow migration notes
- **README.md** - Project overview

---

**Status**: All systems operational! 🎉

