FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal runtime libs for opencv-python on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Which ONNX Runtime wheel to install. These are mutually exclusive drop-in
# replacements for each other, so pick exactly one:
#   onnxruntime          CPU only (default)
#   onnxruntime-openvino Intel CPU / iGPU / NPU via the OpenVINO provider
#   onnxruntime-gpu      NVIDIA CUDA
# Usage: docker build --build-arg ONNXRUNTIME_PACKAGE=onnxruntime-openvino ...
ARG ONNXRUNTIME_PACKAGE=onnxruntime

# Core Python deps (ONNX is now the default backend)
COPY Pipfile Pipfile.lock ./
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        fastapi \
        uvicorn \
        python-multipart \
        pydantic \
        pydantic-settings \
        pillow \
        exceptiongroup \
        numpy \
        "$ONNXRUNTIME_PACKAGE" \
        opencv-python \
        scipy \
        shapely

# Optional: install tensorflow for tflite backend support
# Usage: docker build --build-arg INSTALL_TENSORFLOW=1 ...
ARG INSTALL_TENSORFLOW=0
RUN if [ "$INSTALL_TENSORFLOW" = "1" ]; then python -m pip install tensorflow; fi

# Optional: install moondream (torch + transformers) for VLM backend
# Usage: docker build --build-arg INSTALL_MOONDREAM=1 ...
ARG INSTALL_MOONDREAM=0
RUN if [ "$INSTALL_MOONDREAM" = "1" ]; then \
    python -m pip install \
        "transformers>=4.51.1,<5.0" \
        "torch>=2.7.0" \
        "accelerate>=1.10.0"; \
    fi

COPY . .

# Optionally bake the TFLite model (only useful if tensorflow is installed)
# Usage: docker build --build-arg DOWNLOAD_DEFAULT_MODEL=1 ...
ARG DOWNLOAD_DEFAULT_MODEL=0
RUN if [ "$DOWNLOAD_DEFAULT_MODEL" = "1" ]; then python scripts/download_model.py; fi

EXPOSE 8000

# .env is optional; Unraid can mount it to /app/.env
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
