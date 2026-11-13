# Light Object Detection API

A lightweight Python API for object detection with pluggable backends. This API allows you to detect objects in images using different detection backends, starting with TensorFlow Lite.

The API initializes `DEFAULT_BACKEND` before accepting requests and fails startup
if its model or runtime is unavailable. Other backends initialize on demand;
listing them in `AVAILABLE_BACKENDS` does not require installing every optional
model, runtime, or accelerator. Requests for an optional backend still require its
corresponding model and dependencies. For optional TFLite use, provision the model
with `python scripts/download_model.py` or configure an existing model path.

## Features

- FastAPI-based REST API for object detection
- Pluggable backend architecture for different detection engines
- TensorFlow Lite integration for lightweight, efficient object detection
- Support for image uploads and detection with confidence thresholds
- Extensible design for adding new detection backends

## Requirements

- Python 3.14 for both the Pipfile installation and Docker image
- pipenv (for dependency management)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/opensensor/light-object-detect.git
   cd light-object-detect
   ```

2. Install dependencies:
   ```bash
   pipenv sync
   ```

3. (Optional) Pre-download the default TFLite model:
   ```bash
   pipenv run python scripts/download_model.py
   ```

   If you skip this step, the API will try to download the default model on startup when `DEFAULT_BACKEND=tflite` (requires internet access). Docker builds only download it when `DOWNLOAD_DEFAULT_MODEL=1` is passed.

## Usage

1. Start the API server using the provided script:
   ```bash
   pipenv run python scripts/run_server.py --reload
   ```
   
   Or manually with uvicorn:
   ```bash
   pipenv run uvicorn main:app --reload --host 0.0.0.0 --port 9001
   ```

2. The API will be available at http://localhost:9001

3. Access the API documentation at http://localhost:9001/docs

For access from another computer, use `http://<server-ip>:9001/docs` and
allow TCP port 9001 through the server's firewall. The `--host 0.0.0.0`
option above is required for remote access with Uvicorn.

If installation fails with `No space left on device` while downloading a
package, check the space on the temporary filesystem (`df -h /tmp`). On
systems with a small `/tmp`, use a directory on the disk for temporary files:

```bash
mkdir -p "$HOME/pipenv_tmp"
TMPDIR="$HOME/pipenv_tmp" pipenv sync
```

The base Pipfile installs ONNX Runtime and LiteRT. Ultralytics, PyTorch,
Transformers, and Accelerate are installed for model authoring and Moondream
with `pipenv sync --dev`.
`pipenv sync` uses the committed lockfile without resolving it again. If the
process exits with `Illegal instruction`, try the native imports separately:

```bash
pipenv run python -X faulthandler -c 'import onnxruntime'
pipenv run python -X faulthandler -c 'import ai_edge_litert'
```

After changing runtime packages, refresh the Docker dependency list with
`pipenv lock && pipenv requirements | rg -v '^onnxruntime==' > requirements-runtime.txt`.
The ONNX Runtime wheel is supplied separately by the Docker build argument.

Include the output and `lscpu` details in a bug report. The Pydantic schema warning
was fixed in the project models; it was separate from that process crash.

## Docker (e.g. Unraid / lightNVR)

The default model is an NMS-free YOLO26 nano ONNX export. To compare a larger
YOLO26s model, export it with `nms=False`; the YOLO26 decoder expects
`(1, 300, 6)` detections, while a default raw export needs a different decoder.
The included `yolo26s.onnx` was exported from Ultralytics `yolo26s.pt` with
Ultralytics 8.4.211; its SHA-256 is
`f4c01521140858fe71fccee8b878d984a2332433e4374cbf3ce8f0b93d3f21fa`.

```bash
pipenv sync --dev
pipenv run python scripts/download_models.py --family yolo26 --model-size s
pipenv run python scripts/benchmark_onnx_models.py --images /path/to/camera-frames \
  --model nano=backends/onnx/models/yolo26n-distill.onnx:yolo26 \
  --model small=backends/onnx/models/yolo26s.onnx:yolo26 \
  --model yolo11=backends/onnx/models/yolo11n.onnx:yolo11
```

The script reports end-to-end latency and detection counts. Choose a production
model using labeled camera frames and measured recall, false alarms, and latency;
counts alone cannot establish accuracy. For distant objects, YOLO26's P2 variant
requires training from its YAML architecture on your camera dataset:

```bash
pipenv run yolo detect train model=yolo26n-p2.yaml data=/path/to/camera-data.yaml imgsz=960 epochs=100
```

YOLOE-26 supports promptable classes. Try it as a separate experiment with
`pipenv run python scripts/try_yoloe.py frame.jpg --classes person,package`.
Its segmentation output needs a separate decoder
and is not selectable through the current ONNX backend. Python 3.15 is the newest
stable Python release, but ONNX Runtime does not yet publish Python 3.15 wheels;
3.14 is the newest version this image can run with the default backend.

### Build

```bash
docker build -t light-object-detect:local .
```

The image installs `ai-edge-litert`, but does not bake in a TFLite model by default.
To include the reference model, build with `--build-arg DOWNLOAD_DEFAULT_MODEL=1`.
Set `DEFAULT_BACKEND=tflite` to select LiteRT by
default, or pass `?backend=tflite` for a single request.

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
