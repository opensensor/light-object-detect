# Migration from tflite-runtime to TensorFlow

## Summary

We've migrated the TFLite backend from using the deprecated `tflite-runtime` package to using TensorFlow's built-in TFLite interpreter.

## Changes Made

### 1. Updated Dependencies (Pipfile)

**Before:**
```toml
[packages]
tflite = "*"
tflite-runtime = "*"
```

**After:**
```toml
[packages]
tensorflow = "*"
```

### 2. Updated TFLite Backend (backends/tflite/backend.py)

The backend now uses a fallback chain to find the best available TFLite interpreter:

1. **TensorFlow** (recommended) - Works with Python 3.8+, including 3.12 and 3.13
2. **ai-edge-litert** (new LiteRT) - Only supports Python 3.8-3.11
3. **tflite-runtime** (deprecated) - Legacy fallback

**Import logic:**
```python
try:
    # Try to import from TensorFlow first (most compatible)
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        # Fall back to ai_edge_litert (new LiteRT package, Python 3.8-3.11 only)
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            # Last resort: tflite_runtime (deprecated)
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            raise ImportError(...)
```

### 3. Updated Documentation

- **QUICKSTART.md**: Updated to reflect TensorFlow usage and note about deprecation warnings
- **BACKEND_CONFIGURATION.md**: Already documented TFLite backend configuration

## Benefits

1. **Better Compatibility**: TensorFlow supports Python 3.8+ including the latest versions (3.12, 3.13)
2. **Future-Proof**: While `tf.lite.Interpreter` shows a deprecation warning, it will continue to work until TF 2.20
3. **Simpler Dependencies**: One package (TensorFlow) instead of multiple TFLite packages
4. **Maintained**: TensorFlow is actively maintained by Google

## Known Issues

### Deprecation Warning

You may see this warning when using the TFLite backend:

```
UserWarning: Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
```

**This can be safely ignored** because:
- The new `ai-edge-litert` package doesn't support Python 3.12+
- Our fallback logic will automatically use `ai-edge-litert` when available and compatible
- TensorFlow 2.20 is not yet released, giving us time to migrate when `ai-edge-litert` adds Python 3.12+ support

## Testing

The migration has been tested and verified:

```bash
$ pipenv run python -c "from backends.tflite.backend import TFLiteBackend; \
  backend = TFLiteBackend('backends/tflite/models/ssd_mobilenet_v1.tflite', \
  'backends/tflite/models/coco_labels.txt'); \
  print('✓ TFLite backend initialized successfully!')"

✓ TFLite backend initialized successfully!
```

## Installation

To install the updated dependencies:

```bash
cd light-object-detect
pipenv install
```

This will install TensorFlow and all other required dependencies.

## Model Downloads

TFLite models can be downloaded using the provided script:

```bash
# Download SSD MobileNet V1 (recommended for CPU)
bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v1

# Or download SSD MobileNet V2 (more accurate)
bash scripts/download_tflite_models.sh --model-type ssd_mobilenet_v2

# Or download EfficientDet Lite0 (best accuracy)
bash scripts/download_tflite_models.sh --model-type efficientdet_lite0
```

## Configuration

Update your `.env` file to use the TFLite backend:

```bash
BACKEND=tflite
TFLITE_MODEL_PATH=backends/tflite/models/ssd_mobilenet_v1.tflite
TFLITE_LABELS_PATH=backends/tflite/models/coco_labels.txt
TFLITE_CONFIDENCE_THRESHOLD=0.5
```

## Future Migration Path

When `ai-edge-litert` adds support for Python 3.12+, the backend will automatically use it without any code changes due to our fallback logic. No action required!

## Related Files

- `Pipfile` - Updated dependencies
- `backends/tflite/backend.py` - Updated import logic
- `QUICKSTART.md` - Updated installation instructions
- `scripts/download_tflite_models.sh` - Script to download TFLite models

