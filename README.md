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

3. Download a sample TFLite model:
   ```bash
   pipenv run python scripts/download_model.py
   ```

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

## API Endpoints

- `GET /` - Root endpoint with API information
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
