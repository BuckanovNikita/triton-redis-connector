# Step 04: Drop Trivial Tests

**Agent:** tester
**Depends on:** steps 02, 03

## Goal

Remove ~53 tests that verify framework guarantees or trivial operations. Lower coverage target to 70%.

## Tasks

### 4.1 Model-tools tests

**File:** `packages/triton_model_tools/tests/test_detection.py`
- Delete: `TestDetection.test_rejects_negative_class_id`
- Delete: `TestDetection.test_accepts_confidence_above_one`
- Delete: `TestDetection.test_rejects_confidence_below_zero`
- Delete: `TestImageDetectionResult.test_rejects_negative_inference_time`
- Delete: `TestFolderInferenceResult.test_rejects_negative_total_time`
- Delete: `TestFolderInferenceResult.test_rejects_negative_avg_time`
- If entire file becomes empty, delete it

**File:** `packages/triton_model_tools/tests/test_detection_io.py`
- Delete: `TestCollectImagePaths.test_finds_jpg_and_png`
- Delete: `TestCollectImagePaths.test_sorted_alphabetically`
- Delete: `TestCollectImagePaths.test_raises_on_empty_directory`
- Delete: `TestCollectImagePaths.test_max_count_limits_results`
- Keep: load/save roundtrip tests if they test real parsing logic

**File:** `packages/triton_model_tools/tests/test_registry.py`
- Delete all `TestYoloModelVariantImgsz` pydantic validation tests:
  - `test_int_imgsz_default`, `test_int_imgsz_custom`, `test_tuple_imgsz`, `test_tuple_imgsz_nonsquare`
  - `test_rejects_zero_int`, `test_rejects_negative_int`, `test_rejects_not_divisible_by_32_int`
  - `test_rejects_tuple_with_zero`, `test_rejects_tuple_not_divisible_by_32`, `test_rejects_tuple_with_negative`

**File:** `packages/triton_model_tools/tests/test_dynamic_batching_drift.py`
- Delete: `TestDynamicBatchingDriftUnit` class entirely (synthetic unit tests for infra-dependent feature)
- Keep: `TestDynamicBatchingDriftIntegration` (real integration test with Triton)

**File:** `packages/triton_model_tools/tests/test_dataset.py`
- Already deleted in step 03

### 4.2 Connector tests

**File:** `packages/triton_redis_connector/tests/test_config/test_schema.py`
- Delete ALL pydantic validation tests (~30 tests):
  - All `TestRedisConnectionConfig` tests (invalid_port_too_high, invalid_port_zero, negative_db_rejected, serialization_roundtrip)
  - All `TestSourceConfig` tests (empty_stream_name, empty_consumer_group, empty_event_model, batch_size, block_ms, recovery_interleave, pending_ttl, serialization_roundtrip)
  - All `TestSinkConfig` tests (empty_stream_name, maxlen, serialization_roundtrip)
  - `TestDecoderType.test_invalid_raises`
  - All `TestInferenceConfig` tests (zero_workers, negative_retry, backoff_multiplier, serialization_roundtrip)
  - All `TestPreprocessingConfig` tests (zero_workers, serialization_roundtrip)
  - All `TestQueueConfig` tests (zero_maxsize)
  - All `TestMetricsConfig` tests (zero_poll_interval, serialization_roundtrip)
  - `TestAppConfig.test_full_serialization_roundtrip`
  - `TestNodeConfig` class entirely — `NodeConfig` deleted in step 02 (including pool_size validator tests)

**File:** `packages/triton_redis_connector/tests/test_inference/test_request.py`
- Delete entire file (all tests are trivial field assignment + pydantic validation)

### 4.3 Update coverage config

**File:** `pyproject.toml` (root)
- Change `fail_under = 75` to `fail_under = 70`

## Verification

```bash
uv run pytest
uv run pytest --cov --cov-fail-under=70
```
