"""
Execution provider selection for the ONNX backend.

Kept free of the ``onnxruntime`` import so the selection logic can be exercised
without an ONNX Runtime build installed.
"""
import glob
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"
OPENVINO_PROVIDER = "OpenVINOExecutionProvider"

# Short, config-friendly aliases for the providers we support. Full ONNX Runtime
# provider names ("TensorrtExecutionProvider", ...) are also accepted verbatim.
PROVIDER_ALIASES: Dict[str, str] = {
    "openvino": OPENVINO_PROVIDER,
    "cuda": "CUDAExecutionProvider",
    "cpu": CPU_PROVIDER,
}


def parse_provider_list(value) -> List[str]:
    """
    Normalize the ONNX_EXECUTION_PROVIDERS setting into a list of names.

    Accepts either a comma-separated string ("openvino,cpu") or a sequence.
    """
    if not value:
        return []
    if isinstance(value, str):
        parts: Iterable[str] = value.split(",")
    else:
        parts = value
    return [part.strip() for part in parts if part and part.strip()]


def build_openvino_options(device_type: Optional[str] = None,
                           precision: Optional[str] = None,
                           num_threads: Optional[int] = None,
                           cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Build the provider_options dict for the OpenVINO execution provider.

    Only keys the user actually set are included — OpenVINO picks sensible
    per-device defaults for the rest, and passing an unsupported value (an
    FP16 precision on a CPU device, say) makes session creation fail.

    Args:
        device_type: CPU, GPU, NPU, AUTO, or a HETERO/MULTI/AUTO device list
        precision: FP32, FP16 or ACCURACY
        num_threads: Inference threads for the CPU device
        cache_dir: Directory for compiled model blobs (large startup win on GPU/NPU)

    Returns:
        Dictionary of OpenVINO provider options
    """
    options: Dict[str, Any] = {}
    if device_type:
        options["device_type"] = str(device_type).strip()
    if precision:
        options["precision"] = str(precision).strip().upper()
    if num_threads:
        options["num_of_threads"] = int(num_threads)
    if cache_dir:
        options["cache_dir"] = str(cache_dir)
    return options


def openvino_runtime_info() -> Dict[str, Any]:
    """
    Ask the OpenVINO runtime which devices it can actually see.

    ``OpenVINOExecutionProvider`` appearing in the provider list only says
    OpenVINO is in use, not which device it landed on. A missing OpenCL driver
    or an un-passed /dev/dri shows up as a perfectly healthy-looking provider
    list running entirely on the CPU device, so the device inventory is the
    only way to tell the two apart from outside the container.

    Returns:
        Dictionary with available_devices and their full names, or an error key
    """
    info: Dict[str, Any] = {}

    try:
        import openvino as ov

        core = ov.Core()
        devices = list(core.available_devices)
        names = {}
        for device in devices:
            try:
                names[device] = core.get_property(device, "FULL_DEVICE_NAME")
            except Exception:  # a device can refuse the property
                pass
        info["available_devices"] = devices
        info["device_names"] = names
    except ImportError as exc:
        # The onnxruntime-openvino wheel bundles the OpenVINO C++ runtime but
        # not always the Python bindings, so the execution provider can be fully
        # working while this import fails. Fall through to the evidence below.
        info["error"] = f"openvino package not importable: {exc}"
    except Exception as exc:
        info["error"] = f"openvino device query failed: {exc}"

    # Recorded either way: with no Python bindings this is the only signal there
    # is, and it covers the two things that actually go wrong — the OpenCL
    # driver missing from the image, and /dev/dri missing from the container.
    info["render_nodes"] = sorted(glob.glob("/dev/dri/render*"))
    info["opencl_vendors"] = sorted(glob.glob("/etc/OpenCL/vendors/*.icd"))
    return info


def resolve_providers(requested, available: Sequence[str],
                      openvino_options: Optional[Dict[str, Any]] = None
                      ) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Map a preference order onto the providers this ONNX Runtime build actually has.

    Providers that are not compiled into the installed wheel are skipped with a
    log line rather than raising — that is what lets a single config work across
    a CUDA box, an OpenVINO box and a plain CPU box. CPUExecutionProvider is
    always appended last so there is always a working fallback.

    Args:
        requested: Preference order, as a comma-separated string or a sequence
        available: Result of ``onnxruntime.get_available_providers()``
        openvino_options: Options to attach to the OpenVINO provider, if selected

    Returns:
        Tuple of (providers, provider_options) ready for ``ort.InferenceSession``
    """
    providers: List[str] = []
    provider_options: List[Dict[str, Any]] = []

    for name in parse_provider_list(requested):
        provider = PROVIDER_ALIASES.get(name.lower(), name)
        if provider in providers:
            continue
        if provider not in available:
            logger.info(
                "ONNX: execution provider '%s' is not available in this onnxruntime "
                "build, skipping it", provider
            )
            continue
        providers.append(provider)
        provider_options.append(
            dict(openvino_options or {}) if provider == OPENVINO_PROVIDER else {}
        )

    if CPU_PROVIDER not in providers:
        providers.append(CPU_PROVIDER)
        provider_options.append({})

    return providers, provider_options
