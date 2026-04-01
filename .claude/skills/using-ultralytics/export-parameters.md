# Export Parameters — Full Reference

## Core

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | str | `"torchscript"` | Target format (`"onnx"`, `"engine"`, `"torchscript"`, `"openvino"`, `"coreml"`, `"tflite"`, `"saved_model"`, `"paddle"`, `"ncnn"`) |
| `imgsz` | int/tuple | 640 | Export image size. Int for square, tuple (h, w) for rectangular |
| `half` | bool | False | FP16 quantization |
| `int8` | bool | False | INT8 quantization (requires calibration data) |
| `device` | str/int | None | Export device (`"cpu"`, `0`, `"0,1"`) |
| `batch` | int | 1 | Static batch size |

## ONNX-Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simplify` | bool | True | Simplify ONNX graph with onnxsim |
| `opset` | int | None | ONNX opset version (auto-selected if None) |
| `dynamic` | bool | False | Dynamic axes for variable batch/image size |
| `nms` | bool | False | Add NMS as ONNX operator |

## TensorRT-Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace` | float | None | TensorRT workspace size in GB |
| `dynamic` | bool | False | Enable dynamic batch support |

## End-to-End NMS

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `end2end` | bool | False | Bake NMS into model. Changes output shape from `(B, 4+C, N)` to `(B, max_det, 6)` where each row is `[x1, y1, x2, y2, conf, cls]` |

## Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keras` | bool | False | Use Keras format for TF SavedModel |
| `optimize` | bool | False | Optimize TorchScript for mobile |

## Notes

- `half=True` with `format="onnx"` creates FP16 ONNX model
- `half=True` with `format="engine"` is recommended for GPU (FP16 TensorRT)
- `dynamic=True` with ONNX enables variable batch size at inference time
- `end2end=True` simplifies postprocessing (no manual NMS needed) but limits max detections
- TensorRT engines are GPU-architecture-specific and NOT portable between GPU models
- `imgsz` must be divisible by 32 (YOLO stride)
