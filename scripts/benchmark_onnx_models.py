"""Compare ONNX detector latency on representative local camera frames.

Example:
  pipenv run python scripts/benchmark_onnx_models.py --images frames/ \
    --model yolo26n=backends/onnx/models/yolo26n-distill.onnx:yolo26 \
    --model yolo11n=backends/onnx/models/yolo11n.onnx:yolo11

Counts are diagnostic only; accuracy needs ground-truth annotations.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backends.onnx.backend import ONNXBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True,
                        help="name=path.onnx:model_type")
    parser.add_argument("--labels", default="backends/onnx/models/coco.txt")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png"})
    if not paths:
        parser.error(f"no images found under {args.images}")
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    print("model,provider,images,median_ms,p95_ms,detections_per_pass")
    for spec in args.model:
        try:
            name, rest = spec.split("=", 1)
            model_path, model_type = rest.rsplit(":", 1)
        except ValueError:
            parser.error(f"invalid model spec: {spec}")
        backend = ONNXBackend(model_path, args.labels, model_type=model_type)
        backend.detect(images[0], args.confidence)  # session and graph warmup
        elapsed = []
        counts = []
        for _ in range(args.repeat):
            count = 0
            for image in images:
                start = time.perf_counter()
                count += len(backend.detect(image, args.confidence))
                elapsed.append((time.perf_counter() - start) * 1000)
            counts.append(count)
        ordered = sorted(elapsed)
        p95 = ordered[max(0, int(len(ordered) * 0.95 + 0.9999) - 1)]
        print(f"{name},{backend.providers[0]},{len(images)},"
              f"{statistics.median(elapsed):.1f},{p95:.1f},"
              f"{statistics.mean(counts):.1f}")


if __name__ == "__main__":
    main()
