"""
Tests for ONNX execution provider selection.

Runnable under pytest or directly with ``python3 tests/test_onnx_providers.py``.
The module under test is loaded by path so these run without onnxruntime
installed — importing ``backends.onnx`` would pull in the real runtime.
"""
import importlib.util
import os
import sys

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backends", "onnx", "providers.py",
)
_spec = importlib.util.spec_from_file_location("onnx_providers", _MODULE_PATH)
providers_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(providers_mod)

build_openvino_options = providers_mod.build_openvino_options
parse_provider_list = providers_mod.parse_provider_list
resolve_providers = providers_mod.resolve_providers

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
OPENVINO = "OpenVINOExecutionProvider"


def test_parse_provider_list_accepts_string_and_sequence():
    assert parse_provider_list("cuda, openvino ,cpu") == ["cuda", "openvino", "cpu"]
    assert parse_provider_list(["cuda", "cpu"]) == ["cuda", "cpu"]
    assert parse_provider_list("") == []
    assert parse_provider_list(None) == []


def test_openvino_selected_when_available():
    providers, options = resolve_providers(
        "openvino,cpu", [OPENVINO, CPU], {"device_type": "GPU"}
    )
    assert providers == [OPENVINO, CPU]
    assert options == [{"device_type": "GPU"}, {}]


def test_openvino_skipped_when_wheel_lacks_it():
    """A config naming openvino must still work on a stock onnxruntime build."""
    providers, options = resolve_providers(
        "cuda,openvino,cpu", [CPU], {"device_type": "AUTO"}
    )
    assert providers == [CPU]
    assert options == [{}]


def test_preference_order_is_honoured():
    providers, _ = resolve_providers("openvino,cuda,cpu", [CUDA, OPENVINO, CPU], {})
    assert providers == [OPENVINO, CUDA, CPU]

    providers, _ = resolve_providers("cuda,openvino,cpu", [CUDA, OPENVINO, CPU], {})
    assert providers == [CUDA, OPENVINO, CPU]


def test_cpu_always_appended_last():
    providers, options = resolve_providers("openvino", [OPENVINO, CPU], {})
    assert providers == [OPENVINO, CPU]
    assert len(options) == len(providers)


def test_duplicates_collapse():
    providers, options = resolve_providers(
        "openvino,OpenVINOExecutionProvider,cpu,cpu", [OPENVINO, CPU], {}
    )
    assert providers == [OPENVINO, CPU]
    assert len(options) == 2


def test_full_provider_names_pass_through():
    providers, _ = resolve_providers(
        "TensorrtExecutionProvider,cpu", ["TensorrtExecutionProvider", CPU], {}
    )
    assert providers == ["TensorrtExecutionProvider", CPU]


def test_options_are_not_shared_between_calls():
    """Each session must get its own dict — ORT mutating one must not leak."""
    base = {"device_type": "NPU"}
    providers, options = resolve_providers("openvino,cpu", [OPENVINO, CPU], base)
    options[0]["device_type"] = "CPU"
    assert base == {"device_type": "NPU"}
    assert providers[0] == OPENVINO


def test_empty_request_still_yields_cpu():
    providers, options = resolve_providers("", [CPU], {})
    assert providers == [CPU]
    assert options == [{}]


def test_build_openvino_options_omits_unset_values():
    assert build_openvino_options() == {}
    assert build_openvino_options(device_type=None, precision=None) == {}
    assert build_openvino_options(device_type="GPU") == {"device_type": "GPU"}


def test_build_openvino_options_normalizes():
    options = build_openvino_options(
        device_type=" NPU ", precision="fp16", num_threads="4", cache_dir="/tmp/ov"
    )
    assert options == {
        "device_type": "NPU",
        "precision": "FP16",
        "num_of_threads": 4,
        "cache_dir": "/tmp/ov",
    }


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
