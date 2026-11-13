"""Ensure the Docker dependency list tracks the Pipenv lockfile."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeRequirementsTest(unittest.TestCase):
    def test_versions_match_lockfile(self):
        lock = json.loads((ROOT / "Pipfile.lock").read_text())["default"]
        expected = {name: spec["version"].removeprefix("==")
                    for name, spec in lock.items() if name != "onnxruntime"}
        actual = {}
        for line in (ROOT / "requirements-runtime.txt").read_text().splitlines():
            if not line or line.startswith("-i "):
                continue
            name, value = line.split("==", 1)
            actual[name] = value.split(";", 1)[0].strip()
        self.assertEqual(actual, expected)
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(
            f"ARG ONNXRUNTIME_PACKAGE=onnxruntime{lock['onnxruntime']['version']}",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
