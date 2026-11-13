"""Try promptable YOLOE detection before adding a production decoder.

Requires `pipenv sync --dev`. The first text prompt downloads an encoder.
Example: python scripts/try_yoloe.py image.jpg --classes person,package
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model", default="yoloe-26s-seg.pt")
    parser.add_argument("--classes", required=True,
                        help="Comma-separated class names to detect")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    from ultralytics import YOLOE

    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    if not classes:
        parser.error("--classes must contain at least one name")
    model = YOLOE(args.model)
    model.set_classes(classes)
    prediction = model.predict(args.image, conf=args.confidence, verbose=False)[0]
    detections = []
    for box in prediction.boxes:
        class_id = int(box.cls.item())
        detections.append({
            "label": prediction.names[class_id],
            "confidence": float(box.conf.item()),
            "xyxy": [float(x) for x in box.xyxy[0].tolist()],
        })
    print(json.dumps(detections, indent=2))


if __name__ == "__main__":
    main()
