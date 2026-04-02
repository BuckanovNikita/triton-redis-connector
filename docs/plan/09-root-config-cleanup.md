# Step 09: Root Config Cleanup

**Agent:** implementer
**Depends on:** step 08

## Goal

Make APP_NAME a pure env var, drop model_repository_offline references, verify shipped config is useful.

## Tasks

### 9.1 APP_NAME as environment variable only

**Current:** `AppConfig.app_name` is a pydantic field with Hydra override `${oc.env:APP_NAME,triton-connector}`.

**Change to:**
- Remove `app_name` field from `AppConfig` in `config/schema.py`
- In `main.py`: read `os.environ.get("APP_NAME", "triton-connector")` directly
- Remove `app_name` from hydra `conf/config.yaml`
- Update any code that accesses `config.app_name` to use the env var directly

### 9.2 Drop model_repository_offline references

The project now uses only `model_repository/` with explicit model control mode. Remove ALL references:

**Config files:**
- `.gitignore:54` — remove `model_repository_offline/` pattern
- `pyproject.toml:112` — remove from vulture `paths` list

**Agent instructions:**
- `.claude/agents/architect.md:536,620` — remove staging/offline mentions
- `.claude/agents/implementer.md:24-26` — remove load/unload via move instructions
- `.claude/agents/reviewer.md:52` — remove staging mention
- `.claude/agents/tester.md:45` — remove staging mention

**Documentation:**
- `docs/architecture.md:400` — rewrite to describe explicit model control mode only

Replace "move between model_repository and model_repository_offline" with "use Triton HTTP API to load/unload: `POST /v2/repository/models/{name}/load`"

### 9.3 Verify shipped config

**Files:** `packages/triton_redis_connector/triton_redis_connector/conf/`
- `config.yaml` — verify all defaults are sensible after recovery removal
- `inference/default.yaml` — verify Triton connection defaults
- `preprocessing/default.yaml` — verify decoder/preprocessing defaults
- `metrics/default.yaml` — verify metrics defaults
- `debug/single_thread.yaml` — verify debug config

Remove any stale defaults. Ensure config is minimal but complete.

## Verification

```bash
uv run pre-commit run --all-files
uv run pytest
```
