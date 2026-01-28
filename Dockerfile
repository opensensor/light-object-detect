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

# Install Python deps (keeps this image self-contained for Unraid)
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
        tensorflow \
        onnxruntime==1.23.2 \
        opencv-python \
        scipy \
        shapely

COPY . .

# Bake the default TFLite model into the image (can be disabled at build time).
# Usage: docker build --build-arg DOWNLOAD_DEFAULT_MODEL=0 ...
ARG DOWNLOAD_DEFAULT_MODEL=1
RUN if [ "$DOWNLOAD_DEFAULT_MODEL" = "1" ]; then python scripts/download_model.py; fi

EXPOSE 8000

# .env is optional; Unraid can mount it to /app/.env
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
