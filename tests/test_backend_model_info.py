"""Backend discovery must serialize NumPy-shaped TFLite metadata as JSON."""
import json
import unittest

import numpy as np
from fastapi.encoders import jsonable_encoder

from backends.tflite.backend import TFLiteBackend
from backends.edgetpu.backend import EdgeTPUBackend


class ModelInfoTest(unittest.TestCase):
    def test_numpy_dimensions_are_json_serializable_without_hardware(self):
        for cls in (TFLiteBackend, EdgeTPUBackend):
            with self.subTest(backend=cls.__name__):
                # Metadata requires neither an interpreter nor a physical Coral.
                backend = object.__new__(cls)
                backend.model_path = "model.tflite"
                backend.labels_path = "labels.txt"
                backend.input_shape = np.array([1, 300, 300, 3], dtype=np.int32)
                backend.labels = ["person"]
                backend.confidence_threshold = 0.5
                backend.model_type = "ssd"
                backend.device = "usb"
                backend.requires_int8 = False
                info = backend.get_model_info()
                self.assertEqual(json.loads(json.dumps(info))["input_shape"], [1, 300, 300, 3])
                self.assertEqual(jsonable_encoder(info)["input_shape"], [1, 300, 300, 3])


if __name__ == "__main__":
    unittest.main()
