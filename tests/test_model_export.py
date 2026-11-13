"""Keep YOLO26 exports compatible with the service's end-to-end decoder."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_models import download_yolo_onnx


class ModelExportTest(unittest.TestCase):
    def test_yolo26_export_selects_one_to_one_head(self):
        calls = []

        class FakeYOLO:
            def __init__(self, name):
                calls.append(name)

            def export(self, **options):
                calls.append(options)
                source = Path(directory) / "export.onnx"
                source.write_bytes(b"fixture")
                return str(source)

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=FakeYOLO)}):
                output = download_yolo_onnx("s", directory, "yolo26")
            self.assertEqual(output.read_bytes(), b"fixture")
            self.assertEqual(output.name, "yolo26s.onnx")
            self.assertEqual(calls, ["yolo26s.pt",
                                     {"format": "onnx", "simplify": True, "nms": False}])


if __name__ == "__main__":
    unittest.main()
