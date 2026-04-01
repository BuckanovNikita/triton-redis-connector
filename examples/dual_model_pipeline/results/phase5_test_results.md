# Phase 5: Deterministic Test Results

## Test Environment
- Triton: `triton:8001` (ready, GPU)
- Test image: `coco8/images/train/000000000025.jpg` (640x426)
- Repeat count: 5
- Max coordinate drift: 0.01

## Models Tested

| Model | Resolution | Output Format | Status |
|-------|-----------|---------------|--------|
| yolo11s_960x608 | 960x608 | Standard [1,84,11970] | READY |
| yolo26l | 640x640 | End-to-end [1,300,6] | READY |
| yolo11s_960x608_dali_ensemble | 960x608 | Ensemble | UNAVAILABLE (DALI backend not in container) |

## Even Path (yolo11s_960x608, metal_type=HK)
- Preprocessing: app-side OpenCV + LetterBox
- 2 detections found (both class 23 = giraffe)
  - class=23 conf=0.924 box=[568.9, 90.2, 880.0, 510.1]
  - class=23 conf=0.827 box=[93.7, 507.4, 286.7, 593.9]
- 5/5 deterministic repeats: **PASS** (zero drift)

## Odd Path (yolo26l, metal_type=GK)
- Preprocessing: app-side OpenCV + LetterBox 640x640
- End-to-end model (built-in NMS): output [1,300,6]
- 2 detections found (both class 23 = giraffe)
  - class=23 conf=0.964 box=[384.2, 174.7, 593.8, 464.6]
  - class=23 conf=0.787 box=[50.4, 462.5, 182.3, 519.7]
- 5/5 deterministic repeats: **PASS** (zero drift)

## Output Format Verification
- Even path: all 7 fields present, detections_json valid, metal_type=HK, camera_no=2
- Odd path: all 7 fields present, detections_json valid, metal_type=GK, camera_no=3

## Metric Extraction Verification
| Metric | Type | Event Class | Status |
|--------|------|-------------|--------|
| inspection_received | Counter | SteelInspectionInput | OK |
| even_camera_frame | Counter | EvenCameraFrame | OK |
| odd_camera_frame | Counter | OddCameraFrame | OK |
| inspection_complete | Counter | InspectionResult | OK |
| inspection_inference_ms | Gauge | InspectionResult | OK |
| camera_no_per_minute | GroupBy | EvenCameraFrame/OddCameraFrame | OK |
| metal_type_per_minute | GroupBy | MetadataInferenceInput | OK |

## Results Summary

| Test | Status |
|------|--------|
| even_deterministic | **PASS** |
| odd_deterministic | **PASS** |
| even_output_format | **PASS** |
| odd_output_format | **PASS** |
| wrapper_metrics | **PASS** |

## Note on DALI Ensemble
The DALI backend is not available in this Triton container. The even camera path currently uses app-side preprocessing (OpenCV + LetterBox) targeting `yolo11s_960x608` directly. When deployed with a DALI-enabled Triton, the pipeline's `DaliPreprocessor` sends raw bytes to the DALI ensemble, requiring no code changes — only the config specifying model_name.

## Reference Detections
Saved to `results/reference_detections.json` for future regression testing.
