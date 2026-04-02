# Step 10: Refactor — Drop Linter Ignores

**Agent:** implementer
**Depends on:** step 09

## Goal

Reduce linter ignores from ~262 to ~118 (55% reduction). Fix code instead of suppressing warnings.

## Tasks

### 10.1 `noqa: D107` — Add `__init__` docstrings (45 -> 0)

Add one-line docstrings to all `__init__` methods that currently suppress D107.

**High-value files (post protocols split):**
- `protocols/execution.py` — 6 base class `__init__` methods
- `connectors/redis_stream.py` — 7 classes
- `engine/shipped_processors.py` — 4 classes
- `metrics/aggregation.py` — 4 classes
- `engine/queue_bus.py` — 2 classes
- All others across 22 files

### 10.2 `noqa: D102` — Add method docstrings (31 -> 0)

Add one-line docstrings to methods that currently suppress D102.

**Target files** (providers/ deleted in step 02.7 — ~14 ignores removed automatically):
- `metrics/aggregation.py` (11) — all `metric_name`, `window_seconds`, `extract_key`, `aggregate` methods
- `protocols/execution.py` (6) — `variant` properties on base classes

### 10.3 `noqa: PLC0415` — Fix non-CLI lazy imports (~15 removable)

**Move to top-level in non-CLI files:**
- `engine/app.py` (5): `QueueBus`, `MetricBuffer`, `MetricEmitter`, `redis`, `Path` — these are always available
- `engine/shipped_processors.py` (3): `tritonclient.grpc`, `numpy` — always installed
- `engine/metric_aggregator.py` (1): `MetricEvent` — always available

**Add per-file-ignore for CLI directory** in `pyproject.toml`:
```toml
[tool.ruff.lint.per-file-ignores]
"packages/triton_model_tools/triton_model_tools/cli/*.py" = ["PLC0415"]
```
This removes ~40 ignores from CLI files without code changes.

### 10.4 `noqa: ANN401` — Narrow `Any` types (~8 removable)

**Fix:**
- `engine/queue_bus.py` (3): `put()`, `get()`, `get_nowait()` — use `Envelope[Any]` or a TypeVar
- `engine/metric_aggregator.py` (3): use proper protocol types instead of `Any`
- `protocols/execution.py` (2): `ThreadSink.__call__` and `BaseThreadSink.__call__` — use `object | None`

**Keep (tritonclient has no stubs):**
- `inference/model_registry.py` (4)
- `metrics/triton_collector.py` (2)

### 10.5 `noqa: PLR0913` — Group constructor params (~5 removable)

**Fix where natural:**
- `engine/app.py.__init__` — if many params are config sub-objects, accept `AppConfig` directly
- `metrics/emitter.py.__init__` — accept `MetricsConfig` instead of individual fields

**Keep where genuinely complex** (classmethod factories, multi-concern constructors).

### 10.6 `noqa: BLE001` — Catch specific exceptions (~4 removable)

**Fix in tests:**
- `tests/conftest.py` (2) — catch specific exception types
- `tests/test_pipeline_*.py` — catch specific types

**Keep for tritonclient** (raises undocumented types):
- `model_registry.py`, `triton_collector.py`, `shipped_processors.py`

### 10.7 `type: ignore` / `pyright: ignore` — Fix where possible (~10 removable)

**Fix:**
- `metrics/buffer.py` — `_buffer.append` type mismatch (fix generic type parameter)
- `engine/registry.py` — return type assertions (use proper casts)
- `engine/shipped_processors.py` — `PipelineEvent` arg-type (fix generic)

**Keep (opaque third-party libs):**
- `cv2.imdecode` return type (OpenCV stubs are wrong)
- `turbojpeg.TurboJPEG()` no-untyped-call (no stubs)
- `OmegaConf.to_container` return type

### 10.8 Other noqa codes

Review remaining codes case-by-case:
- `UP046` (deprecated type alias syntax) — fix to modern syntax
- `PLW0108` (unnecessary lambda) — fix to direct expression
- `ARG002` (unused argument) — remove if truly unused, keep if protocol requires it
- `TC001`/`TC002` (runtime needed imports) — keep with comment explaining why
- `PLR0911` (too many returns) — simplify function or keep
- `S603` (subprocess call) — keep (intentional, validated input)

## Expected results

| Category | Before | After | Method |
|----------|--------|-------|--------|
| D107 | 45 | 0 | Add docstrings |
| D102 | 31 | 0 | Add docstrings |
| PLC0415 | 60 | ~5 | Move imports + per-file-ignore |
| ANN401 | 16 | ~8 | Narrow types |
| PLR0913 | 14 | ~9 | Group params |
| BLE001 | 10 | ~6 | Specific exceptions |
| type: ignore | ~30 | ~20 | Fix generics |
| Other | ~56 | ~30 | Case-by-case |
| **TOTAL** | **262** | **~78** | **70% reduction** |

## Verification

```bash
uv run pre-commit run --all-files
uv run pytest
```
