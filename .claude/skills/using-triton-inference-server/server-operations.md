# Triton Server Operations

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
