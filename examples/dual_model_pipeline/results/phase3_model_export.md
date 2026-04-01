# Phase 3: Model Export Results

## yolo26l ONNX Export

- Source: `yolo26l.pt` (ultralytics v8.4.0, 50.7MB)
- Format: ONNX (opset 20, simplified with onnxslim)
- Architecture: YOLO26l — 190 layers, 24,807,420 parameters, 86.4 GFLOPs
- **End-to-end model** (built-in NMS): output shape `[1, 300, 6]` (300 detections x [x1,y1,x2,y2,conf,class_id])
- Location: `model_repository/yolo26l/1/model.onnx` (95.0MB)
- Config: `model_repository/yolo26l/config.pbtxt`

```
Input:  images  [1, 3, 640, 640] FP32
Output: output0 [1, 300, 6] FP32
```

## DALI Ensemble for yolo11s_960x608

### yolo11s_960x608_dali_preprocess
- Backend: DALI
- Input: `INPUT_0` UINT8[-1] (raw image bytes)
- Output: `OUTPUT_0` FP32[3, 608, 960] (preprocessed tensor)
- Pipeline: decode → aspect-preserving resize → pad(114) → BGR→RGB → normalize(/255) → CHW

### yolo11s_960x608_dali_ensemble
- Platform: ensemble
- Steps: DALI preprocess → yolo11s_960x608 inference
- Input: `INPUT_IMAGE` UINT8[-1]
- Output: `OUTPUT_DETECTIONS` FP32[-1]

## default_models.yaml

Added entry:
```yaml
- name: yolo26l
  ultralytics_model: yolo26l.pt
```

## Model Repository Contents

```
model_repository/
  rescale_postprocess/     — Python backend (coordinate rescaling)
  yolo11s/                 — ONNX 640x640, output [1,84,8400] (standard)
  yolo11s_960x608/         — ONNX 960x608, output [1,84,11970] (standard)
  yolo11s_960x608_dali_preprocess/  — DALI backend
  yolo11s_960x608_dali_ensemble/    — Ensemble (DALI + inference)
  yolo26l/                 — ONNX 640x640, output [1,300,6] (end-to-end)
```
