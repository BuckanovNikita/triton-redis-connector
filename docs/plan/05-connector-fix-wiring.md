# Step 05: Connector — Fix Middleware Wiring

**Agent:** implementer
**Depends on:** step 04

## Goal

Fix RetryMiddleware wiring (documented but not implemented) and verify TritonMetricsCollector is always active.

## Tasks

### 5.1 Wire RetryMiddleware into InferenceWorkerFactory

**File:** `packages/triton_redis_connector/triton_redis_connector/inference/worker.py`

Current state (lines 87-89):
```python
def create_processor(self, shared: InferenceConfig) -> InferenceWorker:
    """Create an InferenceWorker with its own gRPC client."""
    return InferenceWorker(grpc_url=shared.grpc_url)
```

Docstring says "wrapped with RetryMiddleware" but code returns bare worker. Fix:

```python
from triton_redis_connector.engine.middleware import RetryMiddleware

def create_processor(self, shared: InferenceConfig) -> RetryMiddleware[TritonRequest, InferenceOutput | InferenceError | None]:
    """Create an InferenceWorker wrapped with RetryMiddleware."""
    worker = InferenceWorker(grpc_url=shared.grpc_url)
    return RetryMiddleware(
        worker,
        max_attempts=shared.max_retry_attempts,
        base_delay_s=shared.retry_base_delay_s,
        backoff_multiplier=shared.retry_backoff_multiplier,
    )
```

### 5.2 Wire TritonMetricsCollector in main.py

DI (providers/) was dropped in step 02.7. TritonMetricsCollector must be created manually in `main.py`.

**File:** `packages/triton_redis_connector/triton_redis_connector/main.py`

Add manual construction:
```python
from triton_redis_connector.metrics.triton_collector import TritonMetricsCollector

collector = TritonMetricsCollector(grpc_url=app_config.inference.grpc_url)
app = Application.build(config=app_config, processors=processors, triton_collector=collector)
```

**Verify in `engine/app.py`:**
- `Application.build()` accepts `triton_collector` parameter
- `Application.run()` calls `triton_collector.start()`
- `Application.shutdown()` calls `triton_collector.stop()`
- `triton_collector` is optional (None if no Triton URL)

### 5.3 Delete ConcurrencyLimitMiddleware (dead code)

**File:** `packages/triton_redis_connector/triton_redis_connector/engine/middleware.py`

`ConcurrencyLimitMiddleware` (lines 62-88) is defined but never instantiated or imported anywhere in the codebase. Delete it.

If `RetryMiddleware` is the only class remaining in `middleware.py`, keep the file. If both are dead, delete the entire file and inline retry logic where needed.

### 5.4 Enforce ModelBundle as the only model definition path

After dropping `NodeConfig` (step 02.4), `ModelBundle` should be the standard way to define models. Add a guard:

**File:** `packages/triton_redis_connector/triton_redis_connector/engine/app.py`

In `Application.build()`, verify that the `processors` list was expanded from `ModelBundle` instances using `expand_bundle()`. If processors are passed as raw callables without a `name` attribute, log a warning suggesting `ModelBundle` usage.

**Update `CLAUDE.md`** — in Architecture section, replace the sentence about `NodeConfig` being optional with:
- "Models are defined exclusively via `ModelBundle`. The `expand_bundle()` function unpacks a bundle into the processor list for `Application.build()`."

## Verification

```bash
uv run pre-commit run --all-files
uv run pytest packages/triton_redis_connector/
```
