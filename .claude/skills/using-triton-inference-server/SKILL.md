---
name: using-triton-inference-server
description: >
  NVIDIA Triton Inference Server reference. Use when writing config.pbtxt files,
  using tritonclient gRPC/HTTP API, building model repository structure, creating
  ensemble models, configuring instance groups, dynamic batching, health checks,
  model management, Python backend models, or querying Triton metrics/statistics.
---

# NVIDIA Triton Inference Server

## tritonclient gRPC API

### Client Setup

```python
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="triton:8001")
```

### Health Checks

```python
client.is_server_live()    # True if server process is running
client.is_server_ready()   # True if server is ready to accept requests
client.is_model_ready("yolo11s")  # True if specific model is loaded
```

### Running Inference

```python
import numpy as np

tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)

input_tensor = grpcclient.InferInput("images", tensor.shape, "FP32")
input_tensor.set_data_from_numpy(tensor)

output = grpcclient.InferRequestedOutput("output0")

result = client.infer(
    model_name="yolo11s",
    inputs=[input_tensor],
    outputs=[output],
    model_version="",    # empty string = latest version
)

output_data = result.as_numpy("output0")  # numpy array
```

### Data Type Strings for InferInput

| String | NumPy dtype |
|--------|-------------|
| `"FP32"` | `np.float32` |
| `"FP16"` | `np.float16` |
| `"INT8"` | `np.int8` |
| `"INT32"` | `np.int32` |
| `"INT64"` | `np.int64` |
| `"UINT8"` | `np.uint8` |
| `"BOOL"` | `np.bool_` |
| `"BYTES"` | object array of bytes |

### Sending Raw Bytes (for DALI backend)

```python
image_bytes = open("image.jpg", "rb").read()
raw = np.frombuffer(image_bytes, dtype=np.uint8).reshape(1, -1)

input_tensor = grpcclient.InferInput("INPUT_0", raw.shape, "UINT8")
input_tensor.set_data_from_numpy(raw)
```

### Model Repository Index

```python
index = client.get_model_repository_index()
for model_info in index:
    # model_info has: .name, .version, .state ("READY" / "UNAVAILABLE")
    pass
```

### Inference Statistics

```python
response = client.get_inference_statistics()
for stat in response.model_stats:
    name = stat.name
    count = stat.inference_count
    success_ns = stat.inference_stats.success.ns
    queue_ns = stat.inference_stats.queue.ns
    compute_ns = stat.inference_stats.compute_infer.ns
```

### Model Metadata

```python
metadata = client.get_model_metadata("yolo11s")
config = client.get_model_config("yolo11s")
```

### Gotchas

- `set_data_from_numpy` does NOT copy — array must stay alive until `infer()` returns
- Shape must include batch dimension if `max_batch_size > 0` in config
- gRPC channel is NOT fork-safe — each process needs its own client
- `tritonclient` has no type stubs — use `# pyright: ignore[reportMissingTypeStubs]`
- Sync: `tritonclient.grpc`, Async: `tritonclient.grpc.aio` — do not mix

---

## HTTP Health Endpoints

```bash
# Liveness (server process running)
curl http://localhost:8000/v2/health/live

# Readiness (models loaded, accepting requests)
curl http://localhost:8000/v2/health/ready

# Model ready
curl http://localhost:8000/v2/models/yolo11s/ready

# Server metadata
curl http://localhost:8000/v2

# Model repository index
curl -X POST http://localhost:8000/v2/repository/index -d {}

# Prometheus metrics
curl http://localhost:8002/metrics
```

---

## config.pbtxt Reference

### ONNX Model

```protobuf
name: "yolo11s"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 1, 3, 640, 640 ]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 1, 84, 8400 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### TensorRT Model

```protobuf
name: "yolo11s_trt"
platform: "tensorrt_plan"
max_batch_size: 0

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 1, 3, 640, 640 ]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### DALI Backend

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

### Python Backend

```protobuf
name: "my_postprocess"
backend: "python"
max_batch_size: 0

input [
  {
    name: "DETECTIONS"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }
]

output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ 1, -1, 6 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

### Ensemble Model

See `./ensemble-patterns.md` for multi-step ensemble configurations.

```protobuf
name: "dali_ensemble"
platform: "ensemble"
max_batch_size: 0

input [
  {
    name: "INPUT_IMAGE"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]

output [
  {
    name: "OUTPUT_DETECTIONS"
    data_type: TYPE_FP32
    dims: [ 1, 84, 8400 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "model_dali_preprocess"
      model_version: -1
      input_map {
        key: "INPUT_0"
        value: "INPUT_IMAGE"
      }
      output_map {
        key: "OUTPUT_0"
        value: "preprocessed_tensor"
      }
    },
    {
      model_name: "yolo11s"
      model_version: -1
      input_map {
        key: "images"
        value: "preprocessed_tensor"
      }
      output_map {
        key: "output0"
        value: "OUTPUT_DETECTIONS"
      }
    }
  ]
}
```

### ONNX with TensorRT Acceleration

ONNX models can use TensorRT as an execution provider without converting to `.plan`.
Triton builds the TRT engine from ONNX on first inference (slow warmup), then caches it.

```protobuf
name: "my_model_trt_opt"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, 3, 640, 640 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1, 84, 8400 ]
  }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator: [{ name: "tensorrt" }]
  }
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### Dynamic Batching

```protobuf
max_batch_size: 8

dynamic_batching {
  preferred_batch_size: [ 2, 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

### config.pbtxt Data Types

| config.pbtxt | Client string | NumPy |
|--------------|---------------|-------|
| `TYPE_FP32` | `"FP32"` | `np.float32` |
| `TYPE_FP16` | `"FP16"` | `np.float16` |
| `TYPE_INT8` | `"INT8"` | `np.int8` |
| `TYPE_INT16` | `"INT16"` | `np.int16` |
| `TYPE_INT32` | `"INT32"` | `np.int32` |
| `TYPE_INT64` | `"INT64"` | `np.int64` |
| `TYPE_UINT8` | `"UINT8"` | `np.uint8` |
| `TYPE_BOOL` | `"BOOL"` | `np.bool_` |

### Platforms and Backends

| Platform/Backend | File Extension | Use Case |
|------------------|----------------|----------|
| `platform: "onnxruntime_onnx"` | `model.onnx` | ONNX Runtime (CPU/GPU) |
| `platform: "tensorrt_plan"` | `model.plan` | TensorRT optimized GPU |
| `backend: "dali"` | `dali.py` / `model.dali` | GPU preprocessing |
| `backend: "python"` | `model.py` | Custom Python logic |
| `platform: "ensemble"` | (none) | Model chaining |

---

## Model Repository Structure

```
model_repository/
  <model_name>/
    config.pbtxt              # Required (strict mode)
    1/                        # Version directory
      model.onnx              # Model file (name depends on platform)
    2/                        # Optional additional versions
      model.onnx
```

- `--strict-model-config=true` requires complete config.pbtxt
- Version directories must be positive integers
- Ensemble models need a version directory but no model file
- Python backend: version dir contains `model.py`
- DALI backend: version dir contains `dali.py`

---

## Python Backend Model

```python
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        """Called once when model is loaded."""
        pass

    def execute(self, requests):
        """Execute inference on batch of requests."""
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, "INPUT_NAME"
            ).as_numpy()

            # Process
            result = input_tensor * 2.0

            output_tensor = pb_utils.Tensor(
                "OUTPUT_NAME",
                result.astype(np.float32),
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output_tensor])
            )
        return responses

    def finalize(self):
        """Called once when model is unloaded."""
        pass
```

---

## Server Launch Options

```bash
tritonserver \
  --model-repository=/models \
  --model-control-mode=poll \          # none | poll | explicit
  --repository-poll-secs=10 \          # poll interval
  --exit-on-error=false \              # keep running if models fail to load
  --strict-model-config=true \         # require complete config.pbtxt
  --log-verbose=1                      # verbose logging
```

| Mode | Description |
|------|-------------|
| `none` | All models loaded at startup, no changes |
| `poll` | Periodically check for new/changed models |
| `explicit` | Models loaded/unloaded via API only |

---

## Model Control API (explicit mode)

When Triton runs with `--model-control-mode=explicit`, models are loaded/unloaded via API:

```bash
# Load a model
curl -X POST http://localhost:8000/v2/repository/models/my_model/load

# Unload a model
curl -X POST http://localhost:8000/v2/repository/models/my_model/unload

# Check if model is ready after loading
curl http://localhost:8000/v2/models/my_model/ready
```

Via gRPC client:
```python
client.load_model("my_model")
client.unload_model("my_model")
```

---

## perf_analyzer

```bash
perf_analyzer \
  -m yolo11s \
  -i grpc \
  -u localhost:8001 \
  --concurrency-range 1 \
  --measurement-interval 5000 \
  --measurement-count 10 \
  --csv
```

Output columns: Concurrency, Inferences/Second, p50/p95/p99 latency (microseconds).
