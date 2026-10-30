"""Startup requires the default engine, without probing unrelated hardware."""
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import main


class StartupTests(unittest.TestCase):
    def setup_settings(self, default='onnx', available=None):
        return patch.multiple(main.settings, DEFAULT_BACKEND=default,
                              AVAILABLE_BACKENDS=available or ['onnx','tflite','opencv','edgetpu','moondream'])

    def test_optional_backends_are_not_loaded_or_downloaded_during_startup(self):
        detector = Mock()
        detector.get_model_info.return_value = {'model_path':'fixture.onnx'}
        with self.setup_settings(), patch.object(main, 'get_backend', return_value=detector) as get, \
                patch.object(main, 'ensure_tflite_ssd_mobilenet_v1') as download:
            with TestClient(main.app) as client:
                self.assertEqual(client.get('/health').status_code, 200)
            get.assert_called_once_with('onnx')
            download.assert_not_called()

    def test_missing_default_model_or_driver_prevents_healthy_startup(self):
        for error in (FileNotFoundError('fixture model'), OSError('fixture driver')):
            with self.subTest(error=type(error).__name__), self.setup_settings('edgetpu'), \
                    patch.object(main, 'get_backend', side_effect=error):
                with self.assertRaises(type(error)):
                    with TestClient(main.app):pass

    def test_disabled_default_prevents_startup(self):
        with self.setup_settings('edgetpu', ['onnx']), patch.object(main, 'get_backend') as get:
            with self.assertRaises(ValueError):
                with TestClient(main.app):pass
            get.assert_not_called()

    def test_tflite_default_is_provisioned_before_initialization(self):
        calls = []
        def download(**kwargs):
            calls.append('download');return Mock(did_download=False)
        def get(name):
            calls.append(name);return Mock(get_model_info=lambda:{})
        with self.setup_settings('tflite'), patch.object(main, 'get_backend', side_effect=get), \
                patch.object(main, 'ensure_tflite_ssd_mobilenet_v1', side_effect=download):
            with TestClient(main.app) as client:
                self.assertEqual(client.get('/health').status_code, 200)
        self.assertEqual(calls, ['download','tflite'])

    def test_default_download_failure_prevents_startup(self):
        with self.setup_settings('tflite'), patch.object(main, 'get_backend') as get, \
                patch.object(main, 'ensure_tflite_ssd_mobilenet_v1', side_effect=OSError('fixture download')):
            with self.assertRaises(OSError):
                with TestClient(main.app):pass
            get.assert_not_called()


if __name__ == '__main__':unittest.main()
