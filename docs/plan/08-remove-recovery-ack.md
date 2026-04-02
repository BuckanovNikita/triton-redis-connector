# Step 08: Remove Recovery/Ack Logic + Model Version Auto-Detect

**Agent:** implementer
**Depends on:** step 07

## Goal

Remove Redis recovery/ack schema and logic (~400 lines deleted). Replace with model version auto-detection using existing `ModelVersionTracker`.

## Tasks

### 8.1 Simplify config/schema.py

**File:** `packages/triton_redis_connector/triton_redis_connector/config/schema.py`

- Delete `CollectorMode` enum (lines 22-27)
- Remove from `SourceConfig` these 6 fields:
  - `collector_mode: CollectorMode`
  - `recovery_enabled: bool`
  - `recovery_interleave_ratio: int`
  - `pending_ttl_s: float`
  - `newest_first_on_recovery: bool`
  - `recovery_max_rps: int`
- Update `config/__init__.py` — remove `CollectorMode` from exports

### 8.2 Simplify connectors/base.py

**File:** `packages/triton_redis_connector/triton_redis_connector/connectors/base.py`

Remove from `InputConnectorProtocol`:
- `read_pending(count)` method
- `acknowledge(message_id)` method

Keep: `read_new(count, block_ms)`

### 8.3 Simplify connectors/consumer_group.py

**File:** `packages/triton_redis_connector/triton_redis_connector/connectors/consumer_group.py`

- `ensure_group_exists()`: remove `mode` parameter, always use `start_id="$"` (FROM_LATEST behavior)
- Delete `read_pending_newest_first()` method entirely
- Delete `acknowledge()` method entirely

### 8.4 Simplify connectors/redis_stream.py

**File:** `packages/triton_redis_connector/triton_redis_connector/connectors/redis_stream.py`

- Delete deprecated `RedisStreamSource` class (~90 lines)
- Delete `RecoveryRedisStreamSource` class (~70 lines)
- Simplify `RedisStreamInputConnector`: remove `read_pending()` and `acknowledge()` methods
- Simplify `TypedRedisStreamSink`:
  - Remove `input_connector` constructor parameter
  - Remove ack logic from `__call__()` (lines where it calls `self._input.acknowledge(fw.message_id)`)
- Keep: `LiveRedisStreamSource` (the clean live-only reader)

### 8.5 Add model version auto-detection

**File:** `packages/triton_redis_connector/triton_redis_connector/inference/model_registry.py`

`ModelVersionTracker` already exists with `get_latest_version(model_name)`. It's not wired into the inference path.

**Option A (preferred):** Wire into `InferenceWorker.__call__()`:
- If `event.model_version == ""`, query `ModelVersionTracker` for latest
- Pass tracker as constructor parameter to `InferenceWorker`

**Option B:** Wire into `InferenceNode`:
- Before sending to Triton, populate model_version from tracker

**Wire lifecycle (manual, no DI — providers/ deleted in step 02.7):**
- Create tracker in `main.py`: `tracker = ModelVersionTracker(grpc_url=...)`
- Pass to `InferenceWorker` constructor
- `Application.run()` must call `tracker.start()`
- `Application.shutdown()` must call `tracker.stop()`

### 8.6 Update hydra config

**File:** `packages/triton_redis_connector/triton_redis_connector/conf/config.yaml`
- Remove recovery-related fields from source config defaults
- Remove `collector_mode` default

### 8.7 Update tests

- `test_connectors/test_redis_stream.py` — delete deprecated `RedisStreamSource` tests
- `test_connectors/test_typed_sources.py` — delete `RecoveryRedisStreamSource` tests + ack behavior tests
- `test_connectors/test_consumer_group.py` — remove `test_read_pending*` and `test_acknowledge*` methods
- `test_config/test_schema.py` — remove any remaining recovery-related tests (most already deleted in step 04)

### 8.8 Update architecture docs

- `docs/architecture.md` — remove mentions of recovery mode, pending messages, ack logic
- `CLAUDE.md` — update shipped processor descriptions if they reference ack/recovery

## Verification

```bash
uv run pre-commit run --all-files
uv run lint-imports
uv run pytest
```
