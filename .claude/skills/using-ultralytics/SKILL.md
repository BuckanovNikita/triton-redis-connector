---
name: using-ultralytics
description: >
  Ultralytics YOLO predict and export reference. Use when running inference with
  ultralytics YOLO models, exporting to ONNX/TensorRT, configuring export
  parameters (imgsz, half, end2end, dynamic), processing prediction results
  (boxes, confidence, class_id), or working with ultralytics CLI. Covers predict
  API, export API, result parsing, and common parameter combinations.
---

# Ultralytics YOLO: Predict & Export

Quick reference for ultralytics `YOLO` class — prediction and export only.

## Loading Models

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")        # pretrained weights (auto-downloads)
model = YOLO("path/to/best.pt")   # custom trained weights
model = YOLO("yolo11s.onnx")      # ONNX model for inference
```

## Predict API

### Basic Usage

```python
results = model.predict(
    source="image.jpg",       # str | Path | np.ndarray | list
    conf=0.25,                # confidence threshold
    iou=0.45,                 # NMS IoU threshold
    imgsz=640,                # int or (h, w) tuple
    verbose=False,            # suppress output
)
```

### Source Types

| Source | Example |
|--------|---------|
| Single image | `"image.jpg"` |
| Directory | `"images/"` |
| Glob | `"images/*.jpg"` |
| NumPy array | `np.ndarray` shape (H, W, 3) BGR uint8 |
| List of arrays | `[img1, img2]` for batch |
| Video | `"video.mp4"` |
| URL | `"https://example.com/image.jpg"` |
| PIL Image | `PIL.Image.open("image.jpg")` |
| Torch tensor | `torch.Tensor` shape (B, 3, H, W) normalized |
| Stream/webcam | `0` (device id) or `"rtsp://..."` |

### Predict Parameters

See `./predict-parameters.md` for the full parameter table.

Key parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | various | required | Input source |
| `conf` | float | 0.25 | Confidence threshold |
| `iou` | float | 0.7 | NMS IoU threshold |
| `imgsz` | int/tuple | 640 | Inference size |
| `half` | bool | False | FP16 inference |
| `device` | str/int | auto | `"cpu"`, `0`, `"0,1"` |
| `max_det` | int | 300 | Max detections per image |
| `classes` | list[int] | None | Filter by class id |
| `agnostic_nms` | bool | False | Class-agnostic NMS |
| `verbose` | bool | True | Print results |
| `stream` | bool | False | Generator mode for video/large batches |
| `save` | bool | False | Save annotated images |
| `save_txt` | bool | False | Save results as .txt |

### Processing Results

```python
results = model.predict("image.jpg", conf=0.25)

for result in results:
    # Bounding boxes
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()        # (N, 4) x1,y1,x2,y2
    xywh = boxes.xywh.cpu().numpy()        # (N, 4) center_x, center_y, w, h
    conf = boxes.conf.cpu().numpy()         # (N,) confidence scores
    cls = boxes.cls.cpu().numpy().astype(int)  # (N,) class indices

    # Class names
    names = result.names                    # {0: "person", 1: "car", ...}

    # Iterate detections
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = names[class_id]
```

### Result Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `result.boxes` | — | Detection boxes container |
| `result.boxes.xyxy` | (N, 4) | Boxes in x1,y1,x2,y2 |
| `result.boxes.xywh` | (N, 4) | Boxes in cx,cy,w,h |
| `result.boxes.xyxyn` | (N, 4) | Normalized xyxy (0..1) |
| `result.boxes.conf` | (N,) | Confidence scores |
| `result.boxes.cls` | (N,) | Class indices |
| `result.boxes.id` | (N,) | Track IDs (if tracking) |
| `result.masks` | — | Segmentation masks (seg models) |
| `result.keypoints` | — | Keypoints (pose models) |
| `result.probs` | — | Classification probabilities (cls models) |
| `result.obb` | — | Oriented bounding boxes (OBB models) |
| `result.orig_img` | np.ndarray | Original image |
| `result.orig_shape` | tuple | Original (H, W) |
| `result.names` | dict | Class index → name mapping |
| `result.speed` | dict | Preprocess/inference/postprocess ms |

### Stream Mode (Memory-Efficient)

For video or large directories, use `stream=True` to get a generator:

```python
for result in model.predict("video.mp4", stream=True):
    # Process one frame at a time — constant memory
    boxes = result.boxes.xyxy.cpu().numpy()
```

Without `stream=True`, all results are loaded into memory at once.

---

## Export API

### Basic Export

```python
model = YOLO("yolo11s.pt")
exported_path = model.export(format="onnx")  # returns path to exported file
```

### Export Formats

| Format | `format=` | File | Notes |
|--------|-----------|------|-------|
| ONNX | `"onnx"` | `.onnx` | Most portable, good for Triton |
| TensorRT | `"engine"` | `.engine` | GPU-optimized, not portable across GPUs |
| TorchScript | `"torchscript"` | `.torchscript` | PyTorch native |
| OpenVINO | `"openvino"` | `_openvino_model/` | Intel hardware |
| CoreML | `"coreml"` | `.mlpackage` | Apple devices |
| TF SavedModel | `"saved_model"` | `_saved_model/` | TensorFlow ecosystem |
| TF Lite | `"tflite"` | `.tflite` | Mobile/edge |
| PaddlePaddle | `"paddle"` | `_paddle_model/` | PaddlePaddle ecosystem |
| NCNN | `"ncnn"` | `_ncnn_model/` | Mobile deployment |

### Export Parameters

See `./export-parameters.md` for the full parameter table.

Key parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | str | `"torchscript"` | Target format |
| `imgsz` | int/tuple | 640 | Export image size |
| `half` | bool | False | FP16 quantization |
| `int8` | bool | False | INT8 quantization |
| `dynamic` | bool | False | Dynamic axes (ONNX) / dynamic batch (TRT) |
| `simplify` | bool | True | Simplify ONNX graph |
| `opset` | int | None | ONNX opset version |
| `batch` | int | 1 | Static batch size |
| `workspace` | float | None | TensorRT workspace (GB) |
| `nms` | bool | False | Add NMS to ONNX model |
| `end2end` | bool | False | End-to-end NMS (baked into model) |
| `device` | str/int | None | Export device |

### Common Export Recipes

```python
model = YOLO("yolo11s.pt")

# ONNX FP32 (portable, good for CPU inference)
model.export(format="onnx", imgsz=640, simplify=True)

# ONNX FP16
model.export(format="onnx", imgsz=640, half=True)

# ONNX dynamic batch
model.export(format="onnx", imgsz=640, dynamic=True)

# ONNX with custom resolution (h, w)
model.export(format="onnx", imgsz=(960, 608))

# ONNX with end-to-end NMS baked in
model.export(format="onnx", imgsz=640, end2end=True)

# TensorRT FP16 (fastest GPU inference)
model.export(format="engine", imgsz=640, half=True)

# TensorRT FP32
model.export(format="engine", imgsz=640, half=False)

# TensorRT with dynamic batching
model.export(format="engine", imgsz=640, half=True, dynamic=True)

# TensorRT with workspace limit
model.export(format="engine", imgsz=640, half=True, workspace=4.0)
```

### Output Shapes

**Standard export** (no end2end):
- Input: `images` — `(batch, 3, H, W)` float32
- Output: `output0` — `(batch, 4+num_classes, num_predictions)` float32
  - First 4 values: xywh (center format)
  - Remaining: class scores

**End-to-end export** (`end2end=True`):
- Input: `images` — `(batch, 3, H, W)` float32
- Output: `output0` — `(batch, max_det, 6)` float32
  - Each detection: `[x1, y1, x2, y2, confidence, class_id]`
  - NMS already applied, pad with zeros

### ONNX Inspection

```python
import onnx

model = onnx.load("yolo11s.onnx")
for inp in model.graph.input:
    name = inp.name
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    dtype = inp.type.tensor_type.elem_type
    # dtype: 1=float32, 10=float16, 6=int32, 7=int64

for out in model.graph.output:
    name = out.name
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
```

---

## Ultralytics CLI

```bash
# Predict
yolo predict model=yolo11s.pt source=image.jpg conf=0.25 imgsz=640

# Export to ONNX
yolo export model=yolo11s.pt format=onnx imgsz=640

# Export to TensorRT
yolo export model=yolo11s.pt format=engine half=True imgsz=640

# Validate (mAP on dataset)
yolo val model=yolo11s.pt data=coco.yaml imgsz=640
```

---

## Model Naming Convention

Ultralytics model names follow the pattern: `yolo{version}{size}.pt`

| Size suffix | Meaning | Params (approx) |
|-------------|---------|-----------------|
| `n` | Nano | ~3M |
| `s` | Small | ~9-11M |
| `m` | Medium | ~20-25M |
| `l` | Large | ~43-50M |
| `x` | Extra-large | ~68-80M |

Examples: `yolo11s.pt`, `yolo12m.pt`, `yolo26l.pt`

---

## Image Size Constraints

- Must be divisible by model stride (32 for YOLO)
- `imgsz=640` → square 640x640
- `imgsz=(960, 608)` → height=960, width=608
- Non-stride-aligned sizes are auto-padded by ultralytics
- Larger size = better accuracy, slower inference
