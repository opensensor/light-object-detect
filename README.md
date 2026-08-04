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
detection option can be set **per stream** from the stream's custom-endpoint field —
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
the model's input — a 60 px person in a 1080p frame arrives at a 640 px YOLO model
as 20 px, below the ~24 px where it reliably fires. Tiled detection crops regions at
a higher effective scale and rotates through them on a fixed inference budget, so
cost stays flat regardless of scene content.

Set it per request with `tiles`, `min_object_px` and `tile_period`, or server-wide
with the `TILE_*` environment variables:

```bash
curl -X POST "http://localhost:8000/api/v1/detect?tiles=4&min_object_px=60&tile_period=1" \
  -F "file=@frame.jpg"
```

See **[docs/TILED_DETECTION.md](docs/TILED_DETECTION.md)** for the full parameter
reference, the matching `.env` variable names, how to choose `min_object_px` and
`tiles`, and the `tile_period` aliasing trap that causes permanent blind spots.

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
├── docs/                   # Reference documentation
│   └── TILED_DETECTION.md  # Tiled detection parameters and tuning
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
