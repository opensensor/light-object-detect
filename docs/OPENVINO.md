# OpenVINO acceleration for the ONNX backend

OpenVINO runs the same ONNX model on Intel hardware — the CPU, the integrated GPU,
and the NPU found on Core Ultra parts — and is typically several times faster than
the plain CPU provider on the same machine.

**OpenVINO is not a separate backend.** It is an ONNX Runtime *execution provider*
inside the existing `onnx` backend, so `DEFAULT_BACKEND` stays `onnx`. You select it
through `ONNX_EXECUTION_PROVIDERS`, not by changing the backend.

## 1. Install the runtime

`onnxruntime-openvino` is a **drop-in replacement** for `onnxruntime`, not an
addition. Having more than one installed breaks the import:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-openvino
```

Confirm the provider registered:

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect 'OpenVINOExecutionProvider' in the list
```

## 2. Configure

All settings live in `.env` (loaded by `config.py` via `pydantic-settings`). Names are
case-sensitive:

| `.env` variable | Default | Meaning |
|---|---|---|
| `ONNX_EXECUTION_PROVIDERS` | `cuda,openvino,cpu` | Preference order. Providers missing from the installed wheel are skipped, so one value is safe across machines. |
| `ONNX_OPENVINO_DEVICE_TYPE` | `AUTO` | `CPU`, `GPU`, `NPU`, `AUTO`, or a `HETERO`/`MULTI`/`AUTO` device list such as `MULTI:GPU,CPU`. |
| `ONNX_OPENVINO_PRECISION` | unset | `FP32`, `FP16` or `ACCURACY`. |
| `ONNX_OPENVINO_NUM_THREADS` | unset | Inference threads. **CPU device only** — ignored on GPU and NPU. |
| `ONNX_OPENVINO_CACHE_DIR` | unset | Directory for compiled model blobs. Strongly recommended for GPU and NPU; see below. |

`ONNX_EXECUTION_PROVIDERS` accepts the short aliases `cuda`, `openvino` and `cpu`,
or full ONNX Runtime provider names (`TensorrtExecutionProvider`, …) verbatim. CPU is
always appended as the final fallback, so it can never resolve to an empty list.

A typical Intel-only configuration:

```dotenv
ONNX_EXECUTION_PROVIDERS=openvino,cpu
ONNX_OPENVINO_DEVICE_TYPE=GPU
ONNX_OPENVINO_CACHE_DIR=/cache/openvino
```

**Consider pinning `GPU` rather than leaving `AUTO`.** `AUTO` silently settles on the
CPU device when the GPU is unreachable, which looks identical to success from the
outside. Pinning makes that case visible in the logs.

## 3. Cache compiled models

Set `ONNX_OPENVINO_CACHE_DIR` if you target GPU or NPU. The first session compiles the
model for the device, which can take tens of seconds; the cache turns every subsequent
start into a load. Point it at a persistent volume in Docker or it is rebuilt on every
container start.

This matters more than it sounds when lightNVR is the caller: its API client has a
hard-coded 10 s timeout, so an uncached GPU compile can blow the deadline on every
container start.

If the directory is unwritable the backend logs a warning and continues without it,
rather than failing to start — an unwritable cache costs startup time, not
availability.

## 4. Docker

Build with both arguments:

```bash
docker build \
  --build-arg ONNXRUNTIME_PACKAGE=onnxruntime-openvino \
  --build-arg INSTALL_INTEL_GPU=1 \
  --build-arg BASE_IMAGE=python:3.11-slim-bookworm .
```

`INSTALL_INTEL_GPU=1` installs `intel-opencl-icd`, the host OpenCL driver OpenVINO
needs to reach `/dev/dri`. The OpenVINO wheel ships the GPU plugin but not this
driver, and **without it the container still runs — OpenVINO just sees no GPU and
silently uses the CPU device.**

That build arg requires a Debian **bookworm** base: trixie dropped `intel-opencl-icd`
from its repositories. The Dockerfile fails the build with a message naming the fix if
you pair `INSTALL_INTEL_GPU=1` with a trixie base. The default `BASE_IMAGE` stays on
the floating tag because the arm64 Coral runtime is a trixie `.deb`; the two are
mutually exclusive, which is harmless in practice since Intel iGPU builds are never
arm64.

Pass the device through at run time:

```yaml
services:
  light-object-detect:
    build:
      context: .
      args:
        ONNXRUNTIME_PACKAGE: onnxruntime-openvino
        INSTALL_INTEL_GPU: "1"
        BASE_IMAGE: python:3.11-slim-bookworm
    environment:
      - ONNX_OPENVINO_DEVICE_TYPE=GPU
      - ONNX_OPENVINO_CACHE_DIR=/cache/openvino
    volumes:
      - openvino-cache:/cache/openvino
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128

volumes:
  openvino-cache:
```

The NPU needs the `intel-driver-compiler-npu` / `intel-level-zero-npu` packages and
`--device /dev/accel` instead.

## 5. Verify it is really on the GPU

**A provider list containing `OpenVINOExecutionProvider` does not mean the GPU is
being used.** A missing OpenCL driver or an un-passed `/dev/dri` produces a perfectly
healthy-looking provider list running entirely on the CPU device. The device inventory
is the only way to tell them apart.

Startup logs both:

```
ONNX: session for backends/onnx/models/yolo11n.onnx using providers ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
ONNX: OpenVINO devices visible: ['CPU', 'GPU']
```

The same detail is available at runtime:

```bash
curl -s http://localhost:8000/api/v1/backends \
  | jq '.backends.onnx.model_info | {providers, openvino}'
```

`model_info.openvino` carries `available_devices`, `device_names`,
`requested_device_type` and `render_nodes`. If `available_devices` is `["CPU"]` while
you asked for `GPU`, you are on the CPU device — faster than the plain CPU provider,
but not the iGPU. The backend logs a warning naming that case at startup, and
distinguishes the two causes: no `/dev/dri/render*` node passed into the container,
versus no OpenCL ICD installed in the image.

**One non-problem to know about.** On Linux you will often see:

```
device_query: unavailable: the onnxruntime-openvino wheel bundles the OpenVINO
runtime without Python bindings; this is expected and does not indicate a GPU problem
```

The `onnxruntime-openvino` wheel deliberately bundles the OpenVINO C++ runtime without
the Python package, so the execution provider is fully working while the device query
cannot run. This is reported separately from `error` precisely because it says nothing
about device availability.

## 6. Fallback behaviour

If OpenVINO cannot compile the model for the requested device — most common when
pinning `NPU` — the backend logs a warning and retries on CPU rather than failing the
whole API. Throughput drops by roughly an order of magnitude while everything keeps
working, so **watch for that warning if performance looks unchanged after switching.**

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OpenVINOExecutionProvider` absent from `get_available_providers()` | wrong wheel, or two ONNX Runtime wheels installed | `pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-openvino` |
| Provider present, `available_devices` is `["CPU"]`, `render_nodes` empty | `/dev/dri` not passed into the container | add `--device /dev/dri/renderD128` |
| Provider present, `available_devices` is `["CPU"]`, render node present | no OpenCL ICD in the image | rebuild with `--build-arg INSTALL_INTEL_GPU=1` |
| Build fails on `intel-opencl-icd` | trixie base image | add `--build-arg BASE_IMAGE=python:3.11-slim-bookworm` |
| First request after start times out | uncached device compile | set `ONNX_OPENVINO_CACHE_DIR` on a persistent volume |
| No speed-up after switching | silent CPU fallback | check the startup warning and `model_info.openvino` |
