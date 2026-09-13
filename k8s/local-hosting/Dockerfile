# Preserve the exact proven application and dependency versions during migration.
FROM ghcr.io/opensensor/light-object-detect@sha256:449ac5873c308404fd4e4210ac1932998041a2eba64a1fd4c75e83ad3b444532
COPY k8s/local-hosting/onnx_backend.py /app/backends/onnx/backend.py
RUN python -c 'from backends.onnx.backend import ONNXBackend; from PIL import Image; b = ONNXBackend("backends/onnx/models/yolo26n.onnx", "backends/onnx/models/coco.txt", model_type="yolo26"); assert b.session.get_session_options().intra_op_num_threads == 2; b.detect(Image.open("test_image.jpg"))'
