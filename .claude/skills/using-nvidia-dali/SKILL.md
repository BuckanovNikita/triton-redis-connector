---
name: using-nvidia-dali
description: >
  NVIDIA DALI reference for GPU image preprocessing pipelines: fn operators (decoders.image, resize, paste, crop, crop_mirror_normalize, color_space_conversion, external_source, expand_dims, cast, pad), letterbox/resize/normalize recipes and Triton DALI backend integration, including known WSL2 and Blackwell limitations. Use when writing or debugging DALI pipelines or DALI models served by Triton.
---

# NVIDIA DALI — Pipeline & Operator Reference

DALI runs image preprocessing on GPU inside the data pipeline, replacing CPU-bound
OpenCV/PIL operations.

## Gotchas

- `fn.paste` positions are normalized `[0, 1]` relative to canvas size, not pixel coordinates
- `fn.paste` with `min_canvas_size` may create a canvas larger than target — always `fn.crop` after
- `fn.pad` pads at end (bottom-right), not centered — use `fn.paste` for centered letterbox
- `fn.crop_mirror_normalize` with `std=[255, 255, 255]` divides by 255 (it's `(x - mean) / std`)
- DALI pipeline is a lazy graph — operators define the graph, `pipe.run()` executes it
- For Triton backend, the file MUST define a function named `pipe` (not `pipeline` or `main`)
- `@autoserialize` is required for Triton DALI backend — serializes pipeline to protobuf
- DALI expressions (data nodes) cannot be used in Python control flow (`if`, `for`) — they are graph nodes, not values
- **WSL2**: DALI crashes in Triton on WSL2 — do NOT use DALI models in Triton when running under WSL. On host, use `device='cpu'` for DALI operators
- **Blackwell GPUs (RTX 5090)**: DALI fails with nvml error in Triton 26.02 — needs 26.04+ with DALI 1.55+

## Imports

```python
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as dmath
from nvidia.dali.pipeline import Pipeline, pipeline_def
```

For Triton DALI backend:
```python
from nvidia.dali.plugin.triton import autoserialize
```

## Pipeline Definition

### Standalone Pipeline (testing/validation)

```python
pipe = Pipeline(batch_size=1, num_threads=4, device_id=0)
with pipe:
    encoded = fn.external_source(device="cpu", name="INPUT_0")
    images = fn.decoders.image(encoded, device="mixed", output_type=types.BGR)
    # ... more operators ...
    pipe.set_outputs(result)

pipe.build()
pipe.feed_input("INPUT_0", [numpy_array])
(output,) = pipe.run()
result = output.as_cpu().as_array()[0]  # numpy array
```

### Triton Backend Pipeline (model.dali / dali.py)

```python
from nvidia.dali.plugin.triton import autoserialize

@autoserialize
@pipeline_def(batch_size=1, num_threads=4, device_id=0)
def pipe():
    encoded = fn.external_source(device="cpu", name="INPUT_0")
    images = fn.decoders.image(encoded, device="mixed", output_type=types.BGR)
    # ... operators ...
    return result  # returned tensor becomes OUTPUT_0
```

`@autoserialize` serializes the pipeline for Triton DALI backend. The file must
be named `dali.py` or `model.dali` in the model version directory.

---

## Core Operators

### Image Decoding

```python
# Decode JPEG/PNG bytes to GPU tensor
# device="mixed": decode on CPU, transfer result to GPU
images = fn.decoders.image(
    encoded,
    device="mixed",
    output_type=types.BGR,    # or types.RGB
)
# Output: GPU tensor (H, W, C) uint8
```

### Peek Image Shape (without decoding)

```python
shapes = fn.peek_image_shape(encoded)  # (H, W, C) as CPU tensor
orig_h = fn.slice(shapes, 0, 1, axes=[0])
orig_w = fn.slice(shapes, 1, 1, axes=[0])
```

### Resize

```python
# Resize to exact dimensions
resized = fn.resize(
    images,
    resize_x=new_w,         # target width (can be DALI expression)
    resize_y=new_h,         # target height (can be DALI expression)
    interp_type=types.INTERP_LINEAR,  # bilinear interpolation
)

# Resize by scale factor
resized = fn.resize(images, resize_x=0.5, resize_y=0.5)

# Resize shortest edge to 640, keep aspect ratio
resized = fn.resize(images, resize_shorter=640)
```

### Padding with fn.paste

Centered padding with fill value (letterbox style):

```python
padded = fn.paste(
    resized,
    paste_x=fn.cast(pad_left, dtype=types.FLOAT) / target_w,  # normalized [0,1]
    paste_y=fn.cast(pad_top, dtype=types.FLOAT) / target_h,   # normalized [0,1]
    ratio=1.0,
    fill_value=(114, 114, 114),          # ultralytics gray
    min_canvas_size=(target_h, target_w),
)
# Crop to exact target size (paste may overshoot)
padded = fn.crop(
    padded,
    crop_h=target_h,
    crop_w=target_w,
    crop_pos_x=0.0,
    crop_pos_y=0.0,
)
```

### Padding with fn.pad

Simpler padding — pads to target shape:

```python
padded = fn.pad(
    resized,
    fill_value=114.0,
    shape=[target_h, target_w, 3],  # target HWC shape
    axes=[0, 1, 2],
)
```

**fn.paste vs fn.pad**: `fn.paste` allows centered positioning (letterbox).
`fn.pad` always pads at the end (bottom-right). Use `fn.paste` for ultralytics-compatible
centered letterbox.

### Color Space Conversion

```python
rgb = fn.color_space_conversion(
    bgr_image,
    image_type=types.BGR,
    output_type=types.RGB,
)
```

### Normalize + Transpose (crop_mirror_normalize)

Combined normalize/transpose in one operator:

```python
normalized = fn.crop_mirror_normalize(
    rgb,
    mean=[0.0, 0.0, 0.0],           # subtract mean
    std=[255.0, 255.0, 255.0],       # divide by std (=normalize /255)
    output_layout="CHW",             # HWC -> CHW transpose
    dtype=types.FLOAT,               # output dtype
)
```

For ImageNet normalization:
```python
normalized = fn.crop_mirror_normalize(
    rgb,
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    output_layout="CHW",
    dtype=types.FLOAT,
)
```

### Add Batch Dimension

```python
batched = fn.expand_dims(normalized, axes=[0])  # (C,H,W) -> (1,C,H,W)
```

### Type Casting

```python
float_val = fn.cast(int_val, dtype=types.FLOAT)
int_val = fn.cast(float_val, dtype=types.INT32)
```

### Math Operations

```python
import nvidia.dali.math as dmath

# Element-wise min/max
scale = dmath.min(scale_h, scale_w)
val = dmath.max(a, b)

# Arithmetic on DALI data nodes
result = a * b + c
quotient = a / b
```

### External Source

Feed data from outside the pipeline:

```python
# In pipeline definition
encoded = fn.external_source(device="cpu", name="INPUT_0")

# When running standalone
pipe.feed_input("INPUT_0", [numpy_array])
```

For Triton backend, `fn.external_source(name="INPUT_0")` maps to the model's
input tensor named `INPUT_0` in config.pbtxt.

---

## Complete Letterbox Pipeline (Ultralytics-Compatible)

```python
@autoserialize
@pipeline_def(batch_size=1, num_threads=4, device_id=0)
def pipe():
    encoded = fn.external_source(device="cpu", name="INPUT_0")
    images = fn.decoders.image(encoded, device="mixed", output_type=types.BGR)

    shapes = fn.peek_image_shape(encoded)
    orig_h = fn.slice(shapes, 0, 1, axes=[0])
    orig_w = fn.slice(shapes, 1, 1, axes=[0])

    target_h, target_w = 640, 640

    scale_h = target_h / fn.cast(orig_h, dtype=types.FLOAT)
    scale_w = target_w / fn.cast(orig_w, dtype=types.FLOAT)
    scale = dmath.min(scale_h, scale_w)

    new_h = fn.cast(fn.cast(orig_h, dtype=types.FLOAT) * scale, dtype=types.INT32)
    new_w = fn.cast(fn.cast(orig_w, dtype=types.FLOAT) * scale, dtype=types.INT32)

    resized = fn.resize(images, resize_x=new_w, resize_y=new_h,
                        interp_type=types.INTERP_LINEAR)

    pad_top = fn.cast((target_h - new_h) / 2, dtype=types.INT32)
    pad_left = fn.cast((target_w - new_w) / 2, dtype=types.INT32)

    padded = fn.paste(
        resized,
        paste_x=fn.cast(pad_left, dtype=types.FLOAT) / target_w,
        paste_y=fn.cast(pad_top, dtype=types.FLOAT) / target_h,
        ratio=1.0,
        fill_value=(114, 114, 114),
        min_canvas_size=(target_h, target_w),
    )
    padded = fn.crop(padded, crop_h=target_h, crop_w=target_w,
                     crop_pos_x=0.0, crop_pos_y=0.0)

    rgb = fn.color_space_conversion(padded, image_type=types.BGR, output_type=types.RGB)

    normalized = fn.crop_mirror_normalize(
        rgb,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        output_layout="CHW",
        dtype=types.FLOAT,
    )

    return fn.expand_dims(normalized, axes=[0])
```

---

## Triton DALI Backend Integration

### config.pbtxt for DALI Model

```protobuf
name: "model_dali_preprocess"
backend: "dali"
max_batch_size: 0

input [
  {
    name: "INPUT_0"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]

output [
  {
    name: "OUTPUT_0"
    data_type: TYPE_FP32
    dims: [ 1, 3, 640, 640 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### Model Repository Layout

```
model_repository/
  model_dali_preprocess/
    config.pbtxt
    1/
      dali.py           # pipeline script with @autoserialize
```

### Sending Raw Bytes to DALI via gRPC Client

```python
import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="triton:8001")

image_bytes = open("image.jpg", "rb").read()
raw_input = np.frombuffer(image_bytes, dtype=np.uint8)
input_array = raw_input.reshape(1, -1)

input_tensor = grpcclient.InferInput("INPUT_0", input_array.shape, "UINT8")
input_tensor.set_data_from_numpy(input_array)

output = grpcclient.InferRequestedOutput("OUTPUT_0")
result = client.infer(model_name="model_dali_preprocess",
                      inputs=[input_tensor], outputs=[output])
preprocessed = result.as_numpy("OUTPUT_0")  # (1, 3, H, W) float32
```

---

## DALI Types Reference

| Type | Description |
|------|-------------|
| `types.FLOAT` | float32 |
| `types.FLOAT16` | float16 |
| `types.UINT8` | unsigned 8-bit int |
| `types.INT32` | signed 32-bit int |
| `types.BGR` | BGR color space |
| `types.RGB` | RGB color space |
| `types.GRAY` | Grayscale |
| `types.INTERP_LINEAR` | Bilinear interpolation |
| `types.INTERP_CUBIC` | Bicubic interpolation |
| `types.INTERP_LANCZOS3` | Lanczos3 interpolation |
| `types.INTERP_NN` | Nearest neighbor |

## Device Placement

- `device="cpu"` — operator runs on CPU
- `device="gpu"` — operator runs on GPU (input must be on GPU)
- `device="mixed"` — CPU input, GPU output (used by `decoders.image`)
- Once data is on GPU, all subsequent operators should run on GPU
- `fn.external_source` is always CPU; use `device="mixed"` decoder to move to GPU
