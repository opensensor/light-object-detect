# Logging Improvements

## Overview

Enhanced logging has been added throughout the light-object-detect API to provide better visibility into detection processing, errors, and performance.

## Changes Made

### 1. Main Application Logging (`main.py`)

**Added**:
- Structured logging configuration with timestamps
- Startup event handler that logs backend availability
- Backend initialization testing on startup

**Output Example**:
```
2025-11-07 11:30:00 - __main__ - INFO - ============================================================
2025-11-07 11:30:00 - __main__ - INFO - Light Object Detection API Starting
2025-11-07 11:30:00 - __main__ - INFO - ============================================================
2025-11-07 11:30:00 - __main__ - INFO - Available backends: ['tflite', 'onnx', 'opencv']
2025-11-07 11:30:00 - __main__ - INFO - Default backend: onnx
2025-11-07 11:30:00 - __main__ - INFO - ✓ TFLITE backend: backends/tflite/models/ssd_mobilenet_v1.tflite
2025-11-07 11:30:00 - __main__ - INFO - ✓ ONNX backend: backends/onnx/models/yolov8n.onnx
2025-11-07 11:30:00 - __main__ - INFO - ✗ OPENCV backend failed: Model not configured
2025-11-07 11:30:00 - __main__ - INFO - ============================================================
```

### 2. Detection Endpoint Logging (`api/v1/endpoints/detection.py`)

**Added logging for**:
- Request parameters (backend, confidence, filename)
- Backend initialization success/failure
- Image upload and processing details
- Detection results (count, timing, detected classes)
- Filter operations (zones, classes, size)
- Response preparation
- Errors with full stack traces

**Log Levels**:
- **INFO**: Request details, detection results, filter operations
- **DEBUG**: Image processing details, bounding box drawing
- **ERROR**: Failures with stack traces

**Output Example**:
```
2025-11-07 11:30:15 - api.v1.endpoints.detection - INFO - Detection request: backend=onnx, confidence=0.5, filename=frame.jpg
2025-11-07 11:30:15 - api.v1.endpoints.detection - DEBUG - Backend onnx initialized successfully
2025-11-07 11:30:15 - api.v1.endpoints.detection - DEBUG - Read 45678 bytes from uploaded file
2025-11-07 11:30:15 - api.v1.endpoints.detection - DEBUG - Image opened: size=(1920, 1080), mode=RGB
2025-11-07 11:30:15 - api.v1.endpoints.detection - DEBUG - Image preprocessed: size=(1920, 1080)
2025-11-07 11:30:15 - api.v1.endpoints.detection - DEBUG - Using confidence threshold: 0.5
2025-11-07 11:30:15 - api.v1.endpoints.detection - INFO - Detection completed: 3 objects found in 125.3ms
2025-11-07 11:30:15 - api.v1.endpoints.detection - INFO - Detected objects: {'person': 2, 'car': 1}
2025-11-07 11:30:15 - api.v1.endpoints.detection - INFO - Response prepared: 3 detections, 127.5ms total
```

## Running with Enhanced Logging

### Development Mode (Verbose)

```bash
# Start with DEBUG level logging
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001 --log-level debug
```

This will show:
- All INFO, DEBUG, and ERROR messages
- Image processing details
- Backend initialization details
- Detailed timing information

### Production Mode (Standard)

```bash
# Start with INFO level logging (default)
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001
```

This will show:
- Request summaries
- Detection results
- Errors
- Performance metrics

### Quiet Mode

```bash
# Start with WARNING level logging
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001 --log-level warning
```

This will only show:
- Warnings
- Errors

## Log Output Interpretation

### Successful Detection

```
INFO - Detection request: backend=onnx, confidence=0.5, filename=camera1.jpg
INFO - Detection completed: 2 objects found in 98.2ms
INFO - Detected objects: {'person': 1, 'car': 1}
INFO - Response prepared: 2 detections, 100.1ms total
```

**Interpretation**: 
- Request received for ONNX backend
- 2 objects detected (1 person, 1 car)
- Detection took 98.2ms
- Total processing time 100.1ms

### Failed Detection (Invalid Image)

```
INFO - Detection request: backend=onnx, confidence=0.5, filename=bad.jpg
ERROR - Image validation/processing failed: cannot identify image file
```

**Interpretation**:
- Request received
- Image file is corrupted or not a valid image format
- Returns 400 Bad Request to client

### Failed Detection (Backend Error)

```
INFO - Detection request: backend=onnx, confidence=0.5, filename=frame.jpg
ERROR - Failed to initialize backend onnx: Model not found at backends/onnx/models/yolov8n.onnx
```

**Interpretation**:
- Request received
- ONNX backend failed to initialize (model file missing)
- Returns 500 Internal Server Error to client

### Detection with Filtering

```
INFO - Detection request: backend=onnx, confidence=0.5, filename=frame.jpg
INFO - Detection completed: 5 objects found in 102.3ms
INFO - Detected objects: {'person': 2, 'car': 2, 'dog': 1}
INFO - Class filtering (person,car): 5 -> 4 detections
INFO - Zone filtering: 4 -> 2 detections
INFO - Response prepared: 2 detections, 105.8ms total
```

**Interpretation**:
- 5 objects initially detected
- Class filter removed 1 object (dog)
- Zone filter removed 2 more objects (outside zones)
- Final result: 2 detections

## Testing the API

Use the provided test script:

```bash
cd light-object-detect
./test_api.sh
```

This will:
1. Check API health
2. List available backends
3. Create a test image
4. Test detection with ONNX backend
5. Test detection with TFLite backend
6. Display results in a readable format

## Monitoring in Production

### View Real-time Logs

```bash
# Follow the logs
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001 | tee api.log
```

### Filter for Errors Only

```bash
# Show only errors
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001 2>&1 | grep ERROR
```

### Monitor Detection Performance

```bash
# Show only detection timing
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001 2>&1 | grep "Detection completed"
```

### Count Detections by Class

```bash
# Show detected object counts
pipenv run uvicorn main:app --host 0.0.0.0 --port 9001 2>&1 | grep "Detected objects"
```

## Log Format

All logs follow this format:

```
TIMESTAMP - LOGGER_NAME - LEVEL - MESSAGE
```

Example:
```
2025-11-07 11:30:15 - api.v1.endpoints.detection - INFO - Detection completed: 3 objects found in 125.3ms
```

Where:
- **TIMESTAMP**: ISO format timestamp
- **LOGGER_NAME**: Python module name
- **LEVEL**: INFO, DEBUG, WARNING, ERROR
- **MESSAGE**: Human-readable message

## Troubleshooting with Logs

### Problem: 400 Bad Request

**Look for**:
```
ERROR - Image validation/processing failed: ...
ERROR - Invalid zone configuration: ...
ERROR - Invalid backend requested: ...
```

**Solution**: Check the error message for details about what's invalid

### Problem: 500 Internal Server Error

**Look for**:
```
ERROR - Failed to initialize backend: ...
ERROR - Detection failed: ...
```

**Solution**: Check backend configuration and model files

### Problem: Slow Detection

**Look for**:
```
INFO - Detection completed: X objects found in XXXXms
```

**Solution**: 
- If >500ms: Consider using TFLite backend or smaller model
- If >1000ms: Check CPU usage, consider GPU acceleration

### Problem: No Objects Detected

**Look for**:
```
INFO - Detection completed: 0 objects found in XXms
```

**Solution**:
- Lower confidence threshold
- Check if objects are in detection zones
- Verify class filters aren't too restrictive

## Benefits

1. **Debugging**: Quickly identify where requests fail
2. **Performance Monitoring**: Track detection times and bottlenecks
3. **Usage Analytics**: See which backends and classes are most used
4. **Error Tracking**: Full stack traces for debugging
5. **Audit Trail**: Complete record of all detection requests

## Next Steps

Consider adding:
- Log aggregation (e.g., ELK stack, Grafana Loki)
- Metrics collection (e.g., Prometheus)
- Alert rules for errors or slow detections
- Request ID tracking for distributed tracing

---

**All logging improvements are now active!** Start the server with `--log-level debug` to see detailed output.

