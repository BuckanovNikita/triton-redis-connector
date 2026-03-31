---
name: check-infra
description: >
  Infrastructure health checker for Docker services. Use when checking if
  Redis, Triton, or RedisInsight are running, diagnosing test hangs,
  debugging Docker service failures, or "/check-infra". Reports service
  status, loaded Triton models, and common failure diagnostics.
---

# Infrastructure Health Check

Check Docker service health for the development stack.

## Services Expected

| Service | Image | Ports | Health Check |
|---------|-------|-------|--------------|
| Redis | redis:7-alpine | 6379 | `redis-cli ping` → PONG |
| Triton | nvcr.io/nvidia/tritonserver:26.02-py3 | 8000 (HTTP), 8001 (gRPC), 8002 (metrics) | `curl http://localhost:8000/v2/health/ready` → 200 |
| RedisInsight | redis/redisinsight:latest | 5540 | Container running |

## Step 0: Detect Environment

```bash
whoami
```

**If username is `vscode`** → you are running INSIDE the devcontainer.
Docker commands (`docker`, `docker compose`) will NOT work from here.
Follow the **Inside Container** path below.

**If username is NOT `vscode`** → you are on the host machine.
Skip to **Step 1** and follow the normal Docker-based checks.

### Inside Container (username = vscode)

Check services directly via their Docker Compose DNS hostnames:

#### Redis
```bash
redis-cli -h redis -p 6379 ping
```
Expected: `PONG`. If connection refused: Redis container is not running.

#### Triton HTTP Health
```bash
curl -sf http://triton:8000/v2/health/ready && echo "OK" || echo "FAIL"
```
Expected: `OK`. If connection refused: Triton container is not running or still loading.

#### Triton Models
```bash
curl -s http://triton:8000/v2/models | python3 -m json.tool
```
List all loaded models and their state (READY vs UNAVAILABLE).

#### If ANY service is unreachable

**STOP immediately.** Do NOT attempt any workarounds. Tell the user:

> I'm running inside the devcontainer and cannot manage Docker services directly.
> The following service(s) are unreachable: [list them].
>
> Please run these commands from your **host terminal** (outside the container):
>
> ```bash
> # Check what's running
> docker compose -f .devcontainer/docker-compose.yml ps
>
> # Start all services
> docker compose -f .devcontainer/docker-compose.yml up -d
>
> # Check logs for a failing service
> docker compose -f .devcontainer/docker-compose.yml logs <service> --tail=50
> ```
>
> After fixing, ask me to re-run `/check-infra`.

**Do NOT proceed** with any other work that depends on infrastructure until the user confirms services are fixed.

After verifying services from inside the container, skip Steps 1-3 and go directly to Step 4 (model repository check) and Step 5 (summary).

---

## Step 1: Check Service Status

> **Only when running OUTSIDE the container (not vscode user).**

```bash
docker compose -f .devcontainer/docker-compose.yml ps
```

Report which services are running, stopped, or missing.

## Step 2: Health Checks

> **Only when running OUTSIDE the container (not vscode user).**

For each running service, verify health:

### Redis
```bash
docker compose -f .devcontainer/docker-compose.yml exec redis redis-cli ping
```
Expected: `PONG`. If it fails: Redis is starting or crashed.

### Triton HTTP Health
```bash
curl -s http://localhost:8000/v2/health/ready
```
Expected: HTTP 200. If it fails: Triton is still loading models (30s start period)
or has a config error.

### Triton Models
```bash
curl -s http://localhost:8000/v2/models | python3 -m json.tool
```
List all loaded models and their state (READY vs UNAVAILABLE).

## Step 3: Diagnose Failures

> **Only when running OUTSIDE the container (not vscode user).**

If Triton is unhealthy or models are UNAVAILABLE:

```bash
docker compose -f .devcontainer/docker-compose.yml logs triton --tail=50
```

### Common Triton Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `model configuration not found` | Missing config.pbtxt | Create config.pbtxt (strict mode requires it) |
| `Internal: unable to load` | config.pbtxt parse error | Check syntax, field names, data types |
| `GPU not available` | nvidia-docker not installed or no GPU | Install nvidia-container-toolkit or run CPU-only |
| `model version X not found` | Missing version directory | Create `1/` directory with model file |
| `--exit-on-error=false` but no models | Empty model_repository | Export models first with `triton-model-export` |

### Common Redis Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Connection refused | Container not running | `docker compose up -d redis` |
| `LOADING Redis is loading` | AOF replay in progress | Wait for loading to complete |

## Step 4: Check Model Repository

```bash
ls -la model_repository/
```

Compare local models vs what Triton reports as loaded. Each model needs:
- `config.pbtxt` (required with `--strict-model-config=true`)
- At least one version directory (`1/`) containing the model file

## Step 5: Summary

Output a status table:

```
| Service      | Status  | Details                     |
|-------------|---------|-------------------------------|
| Redis        | UP      | PONG response OK              |
| Triton       | UP      | 3 models loaded (2 READY)     |
| RedisInsight | UP      | Web UI at localhost:5540      |
```

If services are down and you are OUTSIDE the container, suggest:
```bash
docker compose -f .devcontainer/docker-compose.yml up -d
```

If services are down and you are INSIDE the container (vscode), follow the
"If ANY service is unreachable" instructions from Step 0.
