# light-object-detect Enhancements

This document describes the new features and capabilities added to the light-object-detect API.

## Table of Contents

1. [Multiple Detection Backends](#multiple-detection-backends)
2. [Zone-Based Detection](#zone-based-detection)
3. [Object Tracking](#object-tracking)
4. [Event Filtering](#event-filtering)
5. [Integration with lightNVR](#integration-with-lightnvr)

---

## Multiple Detection Backends

The API now supports multiple detection backends, allowing you to choose the best model for your use case.

### Available Backends

1. **TFLite** (default) - TensorFlow Lite models
2. **ONNX Runtime** - ONNX models including YOLOv8, YOLOv5
3. **OpenCV DNN** - Multiple formats (Darknet/YOLO, Caffe, TensorFlow, ONNX)

### Configuration

Set the backend in your environment variables or config:

```bash
# For ONNX Runtime (YOLOv8)
BACKEND=onnx
ONNX_MODEL_PATH=/path/to/yolov8n.onnx
ONNX_LABELS_PATH=/path/to/coco_labels.txt
ONNX_CONFIDENCE_THRESHOLD=0.5
ONNX_IOU_THRESHOLD=0.45
ONNX_MODEL_TYPE=yolov8

# For OpenCV DNN
BACKEND=opencv
OPENCV_MODEL_PATH=/path/to/yolov4.weights
OPENCV_CONFIG_PATH=/path/to/yolov4.cfg
OPENCV_LABELS_PATH=/path/to/coco_labels.txt
OPENCV_CONFIDENCE_THRESHOLD=0.5
OPENCV_NMS_THRESHOLD=0.4
```

### Usage

```bash
# Detect with ONNX backend
curl -X POST "http://localhost:8000/api/v1/detect?backend=onnx" \
  -F "file=@image.jpg"

# Detect with OpenCV backend
curl -X POST "http://localhost:8000/api/v1/detect?backend=opencv" \
  -F "file=@image.jpg"
```

---

## Zone-Based Detection

Detect objects only in specific regions of the image using polygon zones.

### Features

- Define multiple detection zones with polygon coordinates
- Filter detections by zone
- Per-zone class filtering (e.g., only detect persons in entrance zone)
- Per-zone confidence thresholds
- Three zone modes: `center`, `any`, `all`

### Zone Configuration

```json
{
  "zones": [
    {
      "id": "entrance",
      "name": "Entrance Area",
      "polygon": [
        {"x": 0.0, "y": 0.0},
        {"x": 0.5, "y": 0.0},
        {"x": 0.5, "y": 1.0},
        {"x": 0.0, "y": 1.0}
      ],
      "enabled": true,
      "filter_classes": ["person"],
      "min_confidence": 0.7
    },
    {
      "id": "parking",
      "name": "Parking Lot",
      "polygon": [
        {"x": 0.5, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 0.5, "y": 1.0}
      ],
      "enabled": true,
      "filter_classes": ["car", "truck", "motorcycle"],
      "min_confidence": 0.6
    }
  ],
  "zone_mode": "center"
}
```

### Zone Modes

- **center**: Detection's bounding box center must be in zone
- **any**: Any part of detection's bounding box must overlap with zone
- **all**: Entire detection's bounding box must be within zone

### Usage

```bash
# Detect with zones
ZONES='{"zones":[{"id":"zone1","name":"Area 1","polygon":[{"x":0.0,"y":0.0},{"x":0.5,"y":0.0},{"x":0.5,"y":1.0},{"x":0.0,"y":1.0}],"enabled":true}],"zone_mode":"center"}'

curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "zones=$ZONES"
```

### Response

Detections include `zone_id` field:

```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "bounding_box": {
        "x_min": 0.1,
        "y_min": 0.2,
        "x_max": 0.3,
        "y_max": 0.8
      },
      "track_id": null,
      "zone_id": "entrance"
    }
  ]
}
```

---

## Object Tracking

Track objects across frames with unique IDs using IoU-based matching.

### Features

- Assign unique tracking IDs to objects across frames
- Multi-stream support (separate trackers per stream)
- Configurable tracking parameters
- Track lifecycle management (age, hits, misses)
- Automatic track cleanup for old/lost objects

### Tracking Parameters

- **max_age**: Maximum frames to keep track without updates (default: 30)
- **min_hits**: Minimum hits before track is confirmed (default: 3)
- **iou_threshold**: Minimum IoU for matching (default: 0.3)
- **max_time_since_update**: Maximum time in seconds (default: 1.0)

### Usage

```bash
# Detect and track objects
curl -X POST "http://localhost:8000/api/v1/detect-and-track?stream_name=front_door" \
  -F "file=@frame1.jpg"

curl -X POST "http://localhost:8000/api/v1/detect-and-track?stream_name=front_door" \
  -F "file=@frame2.jpg"

# Get active tracks for a stream
curl "http://localhost:8000/api/v1/tracks/front_door"

# Reset tracker for a stream
curl -X POST "http://localhost:8000/api/v1/tracks/front_door/reset"

# Reset all trackers
curl -X POST "http://localhost:8000/api/v1/tracks/reset-all"
```

### Response

Detections include `track_id` field:

```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "bounding_box": {
        "x_min": 0.1,
        "y_min": 0.2,
        "x_max": 0.3,
        "y_max": 0.8
      },
      "track_id": 1,
      "zone_id": null
    }
  ]
}
```

---

## Event Filtering

Filter detections based on various criteria.

### Filter Types

1. **Class Filtering** - Filter by object class
2. **Confidence Filtering** - Filter by confidence threshold
3. **Size Filtering** - Filter by bounding box size
4. **Preset Filters** - Person-only, vehicle-only, animal-only

### Usage

```bash
# Filter by class (person and car only)
curl -X POST "http://localhost:8000/api/v1/detect?filter_classes=person,car" \
  -F "file=@image.jpg"

# Filter by minimum size
curl -X POST "http://localhost:8000/api/v1/detect?min_width=0.1&min_height=0.1" \
  -F "file=@image.jpg"

# Combine filters
curl -X POST "http://localhost:8000/api/v1/detect?filter_classes=person&min_width=0.05&min_height=0.1" \
  -F "file=@image.jpg"
```

### Advanced Filtering

Use the `EventFilter` model for complex filtering rules:

```python
from models.filter import EventFilter, FilterGroup, FilterRule, FilterOperator

# Person-only filter
filter_config = EventFilter(
    enabled=True,
    person_only=True
)

# Custom filter with rules
filter_config = EventFilter(
    enabled=True,
    filter_groups=[
        FilterGroup(
            logic="AND",
            rules=[
                FilterRule(
                    field="label",
                    operator=FilterOperator.IN,
                    value=["person", "car"]
                ),
                FilterRule(
                    field="confidence",
                    operator=FilterOperator.GREATER_EQUAL,
                    value=0.7
                )
            ]
        )
    ]
)
```

---

## Integration with lightNVR

The lightNVR C codebase has been updated to support all new features.

### Updated Structures

**detection_t** (include/video/detection_result.h):
```c
typedef struct {
    char label[MAX_LABEL_LENGTH];
    float confidence;
    float x, y, width, height;
    int track_id;                  // NEW: Tracking ID (-1 if not tracked)
    char zone_id[MAX_ZONE_ID_LENGTH]; // NEW: Zone ID (empty if not in zone)
} detection_t;
```

### Database Schema

The detections table now includes:
- `track_id INTEGER DEFAULT -1`
- `zone_id TEXT DEFAULT ''`

### API Detection Updates

The `api_detection.c` module now:
- Parses `track_id` and `zone_id` from JSON responses
- Stores tracking and zone information in the database
- Maintains backward compatibility with old API responses

### Example Query

```sql
-- Get all person detections in entrance zone
SELECT * FROM detections 
WHERE label = 'person' 
  AND zone_id = 'entrance' 
  AND timestamp > strftime('%s', 'now', '-1 hour');

-- Get tracking history for a specific track
SELECT * FROM detections 
WHERE track_id = 5 
ORDER BY timestamp ASC;
```

---

## Installation

### Dependencies

Add to `Pipfile`:

```toml
[packages]
onnxruntime = "*"
opencv-python = "*"
scipy = "*"
shapely = "*"
```

Install:

```bash
cd light-object-detect
pipenv install
```

### Download Models

```bash
# Download YOLOv8n ONNX model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

# Download COCO labels
wget https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt
```

---

## Performance Considerations

- **ONNX Runtime**: Fastest with CUDA support, good CPU performance
- **OpenCV DNN**: Good balance, supports many formats
- **TFLite**: Lightweight, good for embedded systems
- **Tracking**: Minimal overhead (~5-10ms per frame)
- **Zone Filtering**: Negligible overhead (<1ms)

---

## Future Enhancements

Potential future improvements:
- DeepSORT tracking with appearance features
- Multi-object tracking metrics (MOTA, MOTP)
- Zone crossing detection and counting
- Heatmap generation from detections
- Real-time streaming support
- Webhook notifications for events
- Detection result caching

