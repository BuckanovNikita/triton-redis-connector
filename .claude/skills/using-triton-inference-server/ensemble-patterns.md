# Triton Ensemble Patterns

## How Ensembles Work

Ensemble models chain multiple Triton models in a single request. Data flows between
steps via named intermediate tensors (internal to the ensemble, not exposed to the client).

The client sends input to the ensemble's top-level inputs and receives the final outputs.
Triton handles routing between steps automatically.

## 2-Step: DALI Preprocess -> Inference

```
Client sends: raw JPEG bytes
Step 1: DALI decodes + preprocesses -> float32 tensor
Step 2: ONNX/TRT model runs inference -> detections
Client receives: raw detection tensor
```

```protobuf
name: "yolo_dali_ensemble"
platform: "ensemble"
max_batch_size: 0

input [
  { name: "INPUT_IMAGE"  data_type: TYPE_UINT8  dims: [ -1 ] }
]
output [
  { name: "OUTPUT_DETECTIONS"  data_type: TYPE_FP32  dims: [ 1, 84, 8400 ] }
]

ensemble_scheduling {
  step [
    {
      model_name: "yolo_dali_preprocess"
      model_version: -1
      input_map  { key: "INPUT_0"  value: "INPUT_IMAGE" }
      output_map { key: "OUTPUT_0" value: "preprocessed_tensor" }
    },
    {
      model_name: "inference_model"
      model_version: -1
      input_map  { key: "images"   value: "preprocessed_tensor" }
      output_map { key: "output0"  value: "OUTPUT_DETECTIONS" }
    }
  ]
}
```

## 3-Step: DALI -> Inference -> NMS

```
Step 1: DALI preprocess
Step 2: Inference model
Step 3: NMS postprocess (Python backend)
```

```protobuf
ensemble_scheduling {
  step [
    {
      model_name: "dali_preprocess"
      model_version: -1
      input_map  { key: "INPUT_0"  value: "INPUT_IMAGE" }
      output_map { key: "OUTPUT_0" value: "preprocessed_tensor" }
    },
    {
      model_name: "inference_model"
      model_version: -1
      input_map  { key: "images"   value: "preprocessed_tensor" }
      output_map { key: "output0"  value: "raw_detections" }
    },
    {
      model_name: "nms_postprocess"
      model_version: -1
      input_map  { key: "INPUT_0"  value: "raw_detections" }
      output_map { key: "OUTPUT_0" value: "OUTPUT_DETECTIONS" }
    }
  ]
}
```

## 3-Step: DALI -> Inference -> Rescale

Rescale maps letterbox coordinates back to original image space.
Client must provide scale and padding metadata alongside the image.

```protobuf
input [
  { name: "INPUT_IMAGE"  data_type: TYPE_UINT8   dims: [ -1 ] },
  { name: "SCALE"        data_type: TYPE_FP32    dims: [ 1 ] },
  { name: "PAD_LEFT"     data_type: TYPE_FP32    dims: [ 1 ] },
  { name: "PAD_TOP"      data_type: TYPE_FP32    dims: [ 1 ] }
]

ensemble_scheduling {
  step [
    {
      model_name: "dali_preprocess"
      model_version: -1
      input_map  { key: "INPUT_0"  value: "INPUT_IMAGE" }
      output_map { key: "OUTPUT_0" value: "preprocessed_tensor" }
    },
    {
      model_name: "inference_model"
      model_version: -1
      input_map  { key: "images"   value: "preprocessed_tensor" }
      output_map { key: "output0"  value: "raw_detections" }
    },
    {
      model_name: "postprocess_model"
      model_version: -1
      input_map {
        key: "DETECTIONS"  value: "raw_detections"
        key: "SCALE"       value: "SCALE"
        key: "PAD_LEFT"    value: "PAD_LEFT"
        key: "PAD_TOP"     value: "PAD_TOP"
      }
      output_map { key: "RESCALED_DETECTIONS" value: "OUTPUT_DETECTIONS" }
    }
  ]
}
```

## 4-Step: DALI -> Inference -> NMS -> Rescale

Full pipeline for non-end2end models that need NMS and coordinate mapping.

```protobuf
ensemble_scheduling {
  step [
    {
      model_name: "dali_preprocess"
      model_version: -1
      input_map  { key: "INPUT_0"  value: "INPUT_IMAGE" }
      output_map { key: "OUTPUT_0" value: "preprocessed_tensor" }
    },
    {
      model_name: "inference_model"
      model_version: -1
      input_map  { key: "images"   value: "preprocessed_tensor" }
      output_map { key: "output0"  value: "raw_detections" }
    },
    {
      model_name: "nms_postprocess"
      model_version: -1
      input_map  { key: "INPUT_0"  value: "raw_detections" }
      output_map { key: "OUTPUT_0" value: "nms_detections" }
    },
    {
      model_name: "postprocess_model"
      model_version: -1
      input_map {
        key: "DETECTIONS"  value: "nms_detections"
        key: "SCALE"       value: "SCALE"
        key: "PAD_LEFT"    value: "PAD_LEFT"
        key: "PAD_TOP"     value: "PAD_TOP"
      }
      output_map { key: "RESCALED_DETECTIONS" value: "OUTPUT_DETECTIONS" }
    }
  ]
}
```

## Key Rules

- `model_version: -1` means latest version
- `input_map` and `output_map` connect step tensors to ensemble-level names or intermediate names
- Intermediate tensor names (like `preprocessed_tensor`, `raw_detections`) are internal — not exposed to client
- Ensemble models need a `1/` version directory but NO model file inside it
- All component models must be in the same model repository
- If any component model is UNAVAILABLE, the ensemble fails to load
