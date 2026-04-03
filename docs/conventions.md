# Code Conventions and Policies

## Code Style

- **Logging**: always `loguru`, never `print`
- **Configs/models**: always `pydantic`
- **Error handling**: never bare `except Exception` — catch specific types. Never silently swallow exceptions.
- **No dynamic access**: avoid `getattr`/`hasattr`/`__dict__`
- **Avoid comments**: improve names and code structure instead
- **Avoid short functions**: inline functions under 4 lines if used fewer than 4 times
- **Imports**: top-level, never lazy — unless the dependency is truly optional (not in pyproject.toml) or it's a CLI subcommand handler in `__main__.py`. All deps listed in pyproject.toml are non-optional and must be imported at module level.
- **Config file sync**: when moving/renaming files, update ALL config paths: mypy `exclude` in `pyproject.toml`, pyright `exclude` in `pyrightconfig.json`, vulture `paths`/`ignore_names`, and import-linter contracts
- **No custom IoU/NMS**: use `torchvision.ops.box_iou` / `torchvision.ops.nms` (torchvision is allowed)
- **No re-export modules**: import from source package directly, never create proxy modules that only re-export
- **Explicit protocol inheritance**: processors/decoders must explicitly inherit protocol classes, not rely on structural typing

## Metrics Policy

**App must be abstracted from domain metrics.** No domain event types (DecodedImage, DetectionResult, etc.) in `engine/app.py`.

- **Counter on event classes**: Declare `_metrics_ = [Counter("name")]` on the event class to count occurrences
- **Gauge on event fields**: Use `Annotated[float, Gauge("name")]` on event fields to extract and emit values
- **GroupBy on event fields**: Use `Annotated[str, GroupBy("name", window_s=60)]` for time-windowed per-key counting
- **Counter/Gauge/GroupBy** live in `protocols.py` (leaf layer) — any module can import them
- **Tracked on wrappers**: Use `Annotated[ProcessorType, Tracked("name")]` for per-call metrics on parallel nodes
- **MetricEvent**: Only 3 fields — `component_name`, `measure`, `value` (no timestamp, no tags)
- **Zero-value guarantee**: `_log_stats` emits all registered Counter names even when idle
- Update metrics table in `README.md` (section "Метрики") when adding new metrics

## Testing Policy

**Only write tests for real logic and edge cases.** Never write tests for:

- **Framework-guaranteed behavior**: frozen pydantic models, enum values matching their literals, StrEnum string coercion, dataclass field assignment
- **Simple construction/defaults**: creating an object and asserting fields equal constructor args or default values
- **Protocol isinstance checks**: static type checkers (mypy/pyright strict) already verify protocol conformance
- **Constant counts/values**: `assert len(REGISTRY) == 3` breaks when items are added and tests nothing
- **Generated file content**: never assert on pbtxt strings, generated scripts, or config file content — only integration tests running Triton on the output can verify correctness
- **Pydantic/framework validation**: field constraints, enum values, frozen models — these are library guarantees
- **Download/network tools**: one-time use utilities, not worth unit testing
- **Simple file I/O**: collect paths, save txt — cover by e2e tests only

E2E means real infrastructure (Triton, Redis), not mocks of infrastructure.

Coverage target: 70% minimum. Exclude simple utility modules from coverage.

**Good tests verify**: validation logic, algorithms, error paths, serialization roundtrips, edge cases, and integration behavior (deploy to Triton + run inference).
