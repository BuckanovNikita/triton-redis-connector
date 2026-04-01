# Phase 2: InferenceWorker image_bytes Support

## Change

`InferenceWorker.__call__` now supports two input modes:

1. **raw_image (existing)**: FP32 preprocessed tensor - for app-side preprocessing pipelines
2. **image_bytes (new)**: UINT8 raw bytes - for DALI ensemble pipelines where Triton handles decoding

## Code Change (inference/worker.py)

```python
# Before:
if event.raw_image is None:
    raise ValueError("InferenceWorker requires raw_image to be set")
input_tensor = grpcclient.InferInput(event.input_name, event.raw_image.shape, "FP32")

# After:
if event.raw_image is not None:
    input_tensor = grpcclient.InferInput(event.input_name, event.raw_image.shape, "FP32")
    input_tensor.set_data_from_numpy(event.raw_image)
elif event.image_bytes is not None:
    data = np.frombuffer(event.image_bytes, dtype=np.uint8)
    input_tensor = grpcclient.InferInput(event.input_name, [len(data)], "UINT8")
    input_tensor.set_data_from_numpy(data)
else:
    raise ValueError("Neither raw_image nor image_bytes set")
```

## Why

The DALI ensemble model in Triton expects raw JPEG/PNG bytes as UINT8 input. The `InferenceInput` pydantic model already has `image_bytes: bytes | None` with a `@model_validator` ensuring exactly one of `raw_image`/`image_bytes` is set. The worker just wasn't handling the bytes path.

## Test Added

`test_image_bytes_inference` - exercises the UINT8 code path through the worker with a fake gRPC module.
