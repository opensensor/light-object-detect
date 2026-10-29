"""Exercise installation across actual filesystem boundaries, without a network."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from utils.model_download import ensure_tflite_ssd_mobilenet_v1, ModelDownloadError


class ModelInstallTest(unittest.TestCase):
    def archive(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as z:
            z.writestr('detect.tflite', b'model-fixture')
        return data.getvalue()

    def test_cross_filesystem_install(self):
        if not Path('/dev/shm').is_dir():
            self.skipTest('requires a separate writable tmpfs')
        with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
            root = Path(directory)
            self.assertNotEqual(root.stat().st_dev, Path(tempfile.gettempdir()).stat().st_dev)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(self.archive())):
                ensure_tflite_ssd_mobilenet_v1(model_path=str(root/'model'), labels_path=str(root/'labels'))
            self.assertEqual((root/'model').read_bytes(), b'model-fixture')
            self.assertFalse(list(root.glob('.light-object-detect-*')))

    def test_failed_refresh_keeps_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'model').write_bytes(b'known-good')
            with patch('urllib.request.urlopen', side_effect=OSError('interrupted')):
                with self.assertRaises(ModelDownloadError):
                    ensure_tflite_ssd_mobilenet_v1(model_path=str(root/'model'), labels_path=str(root/'labels'), force=True)
            self.assertEqual((root/'model').read_bytes(), b'known-good')
            self.assertFalse(list(root.glob('.light-object-detect-*')))
