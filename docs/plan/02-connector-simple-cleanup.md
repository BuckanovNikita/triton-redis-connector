# Step 02: Connector Package — Simple Cleanup

**Agent:** implementer
**Depends on:** step 01

## Goal

Delete dead files and fix antipatterns in `triton_redis_connector` that don't require architectural changes.

## Tasks

### 2.1 Drop `docs_generator.py`

**Delete:** `packages/triton_redis_connector/triton_redis_connector/config/docs_generator.py`
**Update:** `config/__init__.py` — remove `generate_config_docs` and related exports

### 2.2 Drop `hydra_app.py` from config/

**File:** `packages/triton_redis_connector/triton_redis_connector/config/hydra_app.py`

This has two functions: `load_config_from_dict_config()` and `get_default_config_dir()`.
- Check `main.py` — it's the only consumer
- Move both functions inline into `main.py`
- Delete `hydra_app.py`
- Update `config/__init__.py` — remove exports
- Result: `config/` contains only `__init__.py` + `schema.py`

### 2.3 Delete `letterbox.py` re-export

**Delete:** `packages/triton_redis_connector/triton_redis_connector/preprocessing/letterbox.py`

This file is a pure re-export from `triton_shared_models.preprocessing`. 

**Fix all consumers to import from source:**
- Search: `grep -r "from triton_redis_connector.preprocessing.letterbox\|from triton_redis_connector.preprocessing import.*LetterBox\|from triton_redis_connector.preprocessing import.*LETTERBOX\|from triton_redis_connector.preprocessing import.*NORMALIZE"` across packages
- Replace with: `from triton_shared_models.preprocessing import LetterBoxPreprocessor, LETTERBOX_PAD_VALUE, YOLO_NORMALIZE_DIVISOR`
- Update `preprocessing/__init__.py` — remove letterbox re-exports

**Likely files to update:**
- `engine/shipped_processors.py`
- `__init__.py` (public API)
- Note: `providers/preprocessing_provider.py` will be deleted in step 2.7 — no need to update

### 2.4 Drop NodeConfig, NodeVariant, and assembly.py

`NodeConfig` puts `class_path` in config — contradicts the project rule "no class paths in config, all class refs from code". `assembly.py` (`import_class`, `resolve_type_params`) only supports `NodeConfig` and has **zero production imports** (only its own test file).

The source `providers/node_assembly_provider.py` was already deleted; only stale cache/docs reference it.

**Delete files:**
- `packages/triton_redis_connector/triton_redis_connector/engine/assembly.py`
- `packages/triton_redis_connector/tests/test_engine/test_assembly.py`

**Edit `config/schema.py`:**
- Delete `NodeVariant` enum (lines 130-142)
- Delete `_POOL_VARIANTS` constant (line 145)
- Delete `NodeConfig` class (lines 148-171)
- Remove `nodes: list[NodeConfig] = []` from `AppConfig` (line 208)

**Edit `__init__.py`:**
- Remove `NodeConfig` and `NodeVariant` from import and `__all__`

**Update docs (remove all NodeConfig/assembly/node_assembly_provider references):**
- `CLAUDE.md` — remove "Config-Driven Node Assembly" section and `node_assembly_provider.py` mentions
- `docs/architecture.md` — remove NodeConfig paragraph (line 114-127), assembly section
- `.claude/agents/reviewer.md` — remove lines 139-140 (NodeConfig/class_path guidance)
- `.claude/agents/implementer.md` — remove lines 223-224 (NodeConfig YAML reference), line 246 (assembly wrapping)
- `.claude/agents/architect.md` — remove line 607 (assembly.py reference)

### 2.5 Fix decoder.py — explicit protocol inheritance

**File:** `packages/triton_redis_connector/triton_redis_connector/preprocessing/decoder.py`

Currently `OpenCVDecoder` and `TurboJPEGDecoder` use structural typing (duck typing) to satisfy `ImageDecoderProtocol`. Make it explicit:

```python
from triton_redis_connector.preprocessing.protocols import ImageDecoderProtocol

class OpenCVDecoder(ImageDecoderProtocol):
    ...

class TurboJPEGDecoder(ImageDecoderProtocol):
    ...
```

### 2.6 Wire TurboJPEGDecoder as selectable decoder

Currently `DecoderType.TURBOJPEG` exists in config schema but is **never mapped** to `TurboJPEGDecoder` — the code always creates `OpenCVDecoder` regardless of config.

**Fix in `engine/shipped_processors.py`** (or wherever `ClientPreprocessor`/`ClientPreprocessorNode` is created):
- Read `config.preprocessing.decoder` value
- If `DecoderType.TURBOJPEG`, create `TurboJPEGDecoder()` instead of `OpenCVDecoder()`

Simple match statement:
```python
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder, TurboJPEGDecoder

match config.preprocessing.decoder:
    case DecoderType.OPENCV:
        decoder = OpenCVDecoder()
    case DecoderType.TURBOJPEG:
        decoder = TurboJPEGDecoder()
```

### 2.7 Drop entire `providers/` directory (remove dishka DI)

Dishka DI is **not used in production** — `main.py` constructs everything manually. The 4 provider files + scopes are dead code used only in tests.

**Delete directory:** `packages/triton_redis_connector/triton_redis_connector/providers/`

**Files deleted:**
- `providers/__init__.py`
- `providers/config_provider.py`
- `providers/inference_provider.py`
- `providers/metrics_provider.py`
- `providers/preprocessing_provider.py`
- `providers/scopes.py`
- `providers/shipped_factories.py` (if exists)
- `providers/app_factory.py` (if exists)

**Update imports:**
- `__init__.py` — remove all provider re-exports
- Any test file importing from providers — rewrite to construct objects directly

**Update `pyproject.toml`** (package deps):
- Remove `dishka` from dependencies in `packages/triton_redis_connector/pyproject.toml`

**Update architecture docs:**
- `CLAUDE.md` — remove "Config-Driven Node Assembly" section, remove `providers/` from workspace structure, remove `build_application` factory description
- Remove providers from import-linter layer contracts (root `pyproject.toml`)

**Update import-linter contracts** in root `pyproject.toml`:
- Remove `providers` layer from `connector-layers` contract (was: `providers -> connectors | inference | preprocessing | metrics | routers -> engine -> ...`)
- New: `connectors | inference | preprocessing | metrics | routers -> engine -> config | errors -> protocols (leaf)`

## Verification

```bash
uv run pre-commit run --all-files
uv run pytest packages/triton_redis_connector/
```
