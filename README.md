# Light Object Detection API

A lightweight Python API for object detection with pluggable backends. This API allows you to detect objects in images using different detection backends, starting with TensorFlow Lite.

## Features

- FastAPI-based REST API for object detection
- Pluggable backend architecture for different detection engines
- TensorFlow Lite integration for lightweight, efficient object detection
- Support for image uploads and detection with confidence thresholds
- Extensible design for adding new detection backends

## Requirements

- Python 3.9+
- pipenv (for dependency management)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/light-object-detect.git
   cd light-object-detect
   ```

2. Install dependencies:
   ```bash
   pipenv install
   ```

3. (Optional) Pre-download the default TFLite model:
   ```bash
   pipenv run python scripts/download_model.py
   ```

   If you skip this step, the API will try to download the default model on startup when the `tflite` backend is enabled (requires internet access). Docker builds download the default model by default.

## Usage

1. Start the API server using the provided script:
   ```bash
   pipenv run python scripts/run_server.py --reload
   ```
   
   Or manually with uvicorn:
   ```bash
   pipenv run uvicorn main:app --reload --port 9001
   ```

2. The API will be available at http://localhost:9001

3. Access the API documentation at http://localhost:9001/docs

## Docker (e.g. Unraid / lightNVR)

### Build

```bash
docker build -t light-object-detect:local .
```

By default, the image downloads a small reference TFLite model at build time so the `tflite` backend works out of the box.
To disable this, build with `--build-arg DOWNLOAD_DEFAULT_MODEL=0`.

### Run

Option A: without `.env` (uses defaults from `config.py`):

```bash
docker run --rm -p 8000:8000 --name light-object-detect light-object-detect:local
```

Option B: with `.env` (recommended, e.g. for backend/model paths):

```bash
docker run --rm -p 8000:8000 --name light-object-detect \
  -v "$(pwd)/.env:/app/.env:ro" \
  light-object-detect:local
```

PowerShell:

```powershell
docker run --rm -p 8000:8000 --name light-object-detect `
  -v "${PWD}\.env:/app/.env:ro" `
  light-object-detect:local
```

- **Healthcheck**: `GET /health`
- **Swagger UI**: `GET /docs`

### lightNVR Integration

In lightNVR, the API URL is typically:

- `http://<docker-host>:8000/api/v1/detect`

lightNVR passes that URL through verbatim, including any query string, so every
option below can be set **per stream** from the stream's custom-endpoint field —
no lightNVR changes required:

```
http://<docker-host>:8000/api/v1/detect?stream=driveway&filter_classes=person,car&tiles=4&min_object_px=60&tile_period=1
```

Unrecognised parameters are ignored, so a URL written for a newer server stays
safe against an older one.

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint (useful for Docker/Unraid)
- `GET /api/v1/backends` - List available detection backends
- `POST /api/v1/detect` - Detect objects in an uploaded image

### Example: Detect objects in an image

```bash
curl -X POST "http://localhost:9001/api/v1/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg" \
  -F "backend=tflite" \
  -F "confidence_threshold=0.5"
```

### Tiled detection for small objects

Distant objects are often too small to detect once a frame has been scaled down to
the model's input. A 1080p frame reaching a 640 px YOLO model is scaled by 0.333,
so a 60 px person arrives at 20 px — below the ~24 px where the model reliably
fires.

Tiled detection crops regions at a higher effective scale and rotates through them
on a fixed budget. **Per request it runs exactly `tiles` inferences.** Scene content
never changes that number — it only changes which regions get inspected — so cost is
flat whether the frame is empty or a tree is thrashing in the wind.

Tile 0 is always the whole frame, so large and near objects are still caught every
cycle and behaviour never regresses. The remaining `tiles - 1` rotate over the grid
on a clock-derived cursor, which needs no server-side state.

| Parameter | Default | Meaning |
|---|---|---|
| `tiles` | `1` | Inferences per request. `1` disables tiling entirely. |
| `min_object_px` | `60` | Smallest object to resolve, in **source** pixels. Drives crop size. |
| `tile_overlap` | `0.25` | Overlap between adjacent crops, as a fraction of crop size. |
| `tile_period` | `1.0` | Rotation period in seconds. **Must match the rate the caller actually fires at** — see below. |
| `tile_deadline_s` | `7.0` | Stop issuing tiles once this many seconds have elapsed. |
| `stream` | — | Stream name, logged for correlation. |

```bash
curl -X POST "http://localhost:8000/api/v1/detect?tiles=4&min_object_px=60&tile_period=1" \
  -F "file=@frame.jpg"
```

**Choosing `min_object_px`.** This is the only setting that really matters. It is the
smallest thing you want detected, measured in pixels *in the source frame* — so
photograph the scene and measure a person at the distance you care about. Lower
values magnify more but produce more tiles, which lengthens the rotation:

| Frame | `min_object_px=60` | `=40` | `=24` (native) |
|---|---|---|---|
| 1080p | 3 tiles | 4 | 9 |
| 5 MP | 5 tiles | 10 | 25 |
| 4K | 7 tiles | 16 | 32 (capped) |

If the grid exceeds 32 tiles it is truncated and a warning is logged — treat that as
a signal that `min_object_px` is set finer than the frame can sweep in reasonable time.

**Choosing `tiles`.** It is a per-request parameter, so set it per camera rather than
globally. Close-range cameras can run `tiles=1` and still benefit from full-resolution
detection. Spend the budget on the distant views. As a rough guide, on an Intel HD 530
with the OpenVINO provider a 640 px YOLO inference costs ~30 ms, so six cameras at 1 Hz
and `tiles=4` is around 70% of that GPU. Passing a `tiles` larger than the grid is
harmless — the extra budget is simply not used.

**`tile_period` must match your real detection rate.** The rotation cursor is derived
from the clock, so a caller whose period is an exact multiple of `tile_period` will
sample the same tiles forever and never visit the others — permanent blind spots, not
merely a slower sweep. This matters because lightNVR's keyframe-gated path fires at the
GOP length, not at the configured `detection_interval`. If a stream is gated on
keyframes with a 2 s GOP, set `tile_period=2`.

If the deadline is reached, remaining tiles are skipped and partial results returned.
Tile 0 always runs first, so a slow or degraded host degrades to ordinary whole-frame
behaviour rather than returning nothing.

## Adding New Backends

To add a new detection backend:

1. Create a new module in the `backends` directory
2. Implement the `DetectionBackend` interface
3. Register the backend in `backends/factory.py`

## Project Structure

```
light-object-detect/
├── api/                    # API endpoints
│   ├── v1/                 # API version 1
│   │   └── endpoints/      # API endpoints
│   │       └── detection.py # Detection endpoints
│   └── router.py           # API router
├── backends/               # Detection backends
│   ├── base.py             # Base backend interface
│   ├── factory.py          # Backend factory
│   └── tflite/             # TFLite backend
│       └── backend.py      # TFLite implementation
├── models/                 # Data models
│   └── detection.py        # Detection models
├── scripts/                # Utility scripts
│   ├── download_model.py   # Script to download models
│   ├── run_server.py       # Script to run the API server
│   └── test_api.py         # Script to test the API
├── utils/                  # Utility functions
│   └── image.py            # Image processing utilities
├── config.py               # Application configuration
├── main.py                 # FastAPI application
├── Pipfile                 # Dependencies
└── README.md               # This file
```

## License

Licensed under GPLv3
