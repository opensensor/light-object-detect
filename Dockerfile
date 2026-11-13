# Python 3.14 is the newest release supported by ONNX Runtime wheels.
# The arm64 Coral runtime below targets Debian trixie.
ARG BASE_IMAGE=python:3.14.10-slim-trixie
FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal runtime libs for opencv-python on slim images + feranick's maintained
# Coral Edge TPU runtime for Debian trixie (arm64)
# See: https://github.com/feranick/libedgetpu/releases
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        wget \
        libusb-1.0-0 \
    && if [ "${TARGETARCH}" = "arm64" ]; then \
        wget -q https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.trixie_arm64.deb; \
        dpkg -i libedgetpu1-std_16.0tf2.19.1-1.trixie_arm64.deb; \
        rm libedgetpu1-std_16.0tf2.19.1-1.trixie_arm64.deb; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Which ONNX Runtime wheel to install. These are mutually exclusive drop-in
# replacements for each other, so pick exactly one:
#   onnxruntime          CPU only (default)
#   onnxruntime-openvino Intel CPU / iGPU / NPU via the OpenVINO provider
#   onnxruntime-gpu      NVIDIA CUDA
# Usage: docker build --build-arg ONNXRUNTIME_PACKAGE=onnxruntime-openvino ...
ARG ONNXRUNTIME_PACKAGE=onnxruntime==1.31.1

# Core Python deps (ONNX is now the default backend)
COPY requirements-runtime.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.txt "$ONNXRUNTIME_PACKAGE"

# Optional: Intel GPU userspace driver, required for OpenVINO's GPU device.
# The OpenVINO wheel ships the GPU plugin but still needs the host OpenCL
# driver to see /dev/dri; without it OpenVINO silently falls back to CPU.
# Usage: docker build --build-arg INSTALL_INTEL_GPU=1 ...
ARG INSTALL_INTEL_GPU=0
RUN if [ "$INSTALL_INTEL_GPU" = "1" ]; then \
        apt-get update \
        && { apt-get install -y --no-install-recommends intel-opencl-icd clinfo \
             || { echo "ERROR: intel-opencl-icd is not in this base image's repos."; \
                  echo "Debian trixie dropped it — build with"; \
                  echo "  --build-arg BASE_IMAGE=python:3.13.18-slim-bookworm"; \
                  exit 1; }; } \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Optional: install moondream (torch + transformers) for VLM backend
# Usage: docker build --build-arg INSTALL_MOONDREAM=1 ...
ARG INSTALL_MOONDREAM=0
RUN if [ "$INSTALL_MOONDREAM" = "1" ]; then \
    python -m pip install \
        "transformers==4.57.6" \
        "torch==2.15.0" \
        "accelerate==1.15.0"; \
    fi

COPY . .

# Optionally bake the TFLite model for LiteRT
# Usage: docker build --build-arg DOWNLOAD_DEFAULT_MODEL=1 ...
ARG DOWNLOAD_DEFAULT_MODEL=0
RUN if [ "$DOWNLOAD_DEFAULT_MODEL" = "1" ]; then python scripts/download_model.py; fi

EXPOSE 8000

# .env is optional; Unraid can mount it to /app/.env
# Public Sentry ingestion key; upload credentials are never part of the image.
ARG SENTRY_RELEASE
ENV SENTRY_DSN=https://8bd44d815901211012fcae52d51371f9@o4512337837686784.ingest.us.sentry.io/4512337894703104 \
    SENTRY_ENVIRONMENT=production \
    SENTRY_RELEASE=${SENTRY_RELEASE}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# The hosting image uses the same Python and dependencies as the published
# runtime, with its CPU quota settings and the verified distilled model.
FROM base AS hosting
ENV ONNX_INTRA_OP_THREADS=2
RUN python -c 'from backends.onnx.backend import ONNXBackend; from PIL import Image; b = ONNXBackend("backends/onnx/models/yolo26n-distill.onnx", "backends/onnx/models/coco.txt", model_type="yolo26"); assert b.session.get_session_options().intra_op_num_threads == 2; assert [o.shape for o in b.session.get_outputs()] == [[1, 300, 6]]; b.detect(Image.open("test_image.jpg"))'

# Keep the ordinary runtime as the default docker build target.
FROM base AS runtime
