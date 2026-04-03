# Triton Version Management

Project supports multiple NVIDIA Triton versions via `.env` file at repo root and uv extras.

## How it works

1. **`.env`** at repo root sets `TRITON_VERSION` (e.g., `25.03` or `26.02`)
2. **Docker compose** reads `${TRITON_VERSION}` for container image tags
3. **Root `pyproject.toml`** has conflicting extras `triton-25-03` and `triton-26-02`
4. **`uv sync --extra triton-26-02`** selects which version to install

## Version matrix

| Component | 25.03 | 26.02 |
|---|---|---|
| Triton Server | `25.03-py3` | `26.02-py3` |
| CUDA (in container) | 12.8.1 | 13.1.1 |
| TensorRT (in container) | 10.9.0 | 10.15.1 |
| tritonclient | 2.55.x | 2.66.x |
| numpy | 1.26.x | 2.4.x |
| torch | 2.6.x | 2.11.x |
| torchvision | 0.21.x | 0.26.x |
| DALI (in container) | 1.47 | 1.53 |
| ONNX Runtime | 1.21.0 | 1.21.0 |
| Python | 3.12 | 3.12 |

## Switching versions

```bash
# 1. Edit .env
echo TRITON_VERSION=25.03 > .env

# 2. Re-lock and sync with the matching extra
uv lock && uv sync --all-groups --extra triton-25-03

# 3. Restart containers
docker compose -f .devcontainer/docker-compose.debug.yml down
docker compose -f .devcontainer/docker-compose.debug.yml up -d
```

## Notes for agents

- When implementing features, check the active Triton version in `.env`
- Base dependency pins are wide (`tritonclient>=2.55.0,<3.0.0`, `numpy>=1.26.0,<3.0.0`) to support both
- Extras pin exact version ranges per Triton release
- RTX 5090 (sm_120) requires torch 2.7+ — only works with triton-26-02 extra
- tritonclient gRPC API is stable across all supported versions (v2 protocol)
