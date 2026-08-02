FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal runtime libs for opencv-python on slim images + feranick's maintained
# Coral Edge TPU runtime for Debian trixie (arm64)
# See: https://github.com/feranick/libedgetpu/releases
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        wget \
        libusb-1.0-0 \
    && wget -q https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.trixie_arm64.deb \
    && dpkg -i libedgetpu1-std_16.0tf2.19.1-1.trixie_arm64.deb \
    && rm libedgetpu1-std_16.0tf2.19.1-1.trixie_arm64.deb \
    && rm -rf /var/lib/apt/lists/*

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
        "numpy<2.0" \
        onnxruntime \
        opencv-python \
        scipy \
        shapely

# Install tflite-runtime for Coral Edge TPU support
# Compatible with feranick's libedgetpu 16.0TF2.19.1-1
RUN pip install tflite-runtime==2.14.0

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
