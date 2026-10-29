# Preserve the exact proven application and dependency versions during migration.
FROM ghcr.io/opensensor/light-object-detect@sha256:449ac5873c308404fd4e4210ac1932998041a2eba64a1fd4c75e83ad3b444532
COPY k8s/local-hosting/onnx_backend.py /app/backends/onnx/backend.py
# The pinned base predates the distilled weights, so carry only the model file
# across rather than rebuilding the application image and moving dependencies.
COPY backends/onnx/models/yolo26n-distill.onnx /app/backends/onnx/models/yolo26n-distill.onnx
RUN python -c 'from backends.onnx.backend import ONNXBackend; from PIL import Image; b = ONNXBackend("backends/onnx/models/yolo26n-distill.onnx", "backends/onnx/models/coco.txt", model_type="yolo26"); assert b.session.get_session_options().intra_op_num_threads == 2; meta = b.session.get_modelmeta().custom_metadata_map; assert meta["end2end"] == "True", meta["end2end"]; assert [o.shape for o in b.session.get_outputs()] == [[1, 300, 6]], b.session.get_outputs(); b.detect(Image.open("test_image.jpg"))'

RUN python -m pip install --no-cache-dir sentry-sdk==2.73.0
COPY observability.py main.py /app/
COPY utils/model_download.py /app/utils/model_download.py
ARG SENTRY_RELEASE
ENV SENTRY_DSN=https://8bd44d815901211012fcae52d51371f9@o4512337837686784.ingest.us.sentry.io/4512337894703104 \
    SENTRY_ENVIRONMENT=production \
    SENTRY_RELEASE=${SENTRY_RELEASE}
