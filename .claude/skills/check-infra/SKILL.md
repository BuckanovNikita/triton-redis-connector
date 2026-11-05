---
name: check-infra
description: >
  Checks health of the triton-redis-connector Docker stack (Redis, Triton, RedisInsight, Triton SDK) from the host or from inside the devcontainer, lists loaded Triton models and diagnoses common failures. Use when tests hang, Redis or Triton is unreachable, Triton models are UNAVAILABLE, or the user runs /check-infra.
---

# Infrastructure Health Check

Check Docker service health for the development stack.

## Compose Files

| File | Purpose | Triton health check | Ports exposed |
|------|---------|---------------------|---------------|
| `docker-compose.yml` | Devcontainer (default) | `/v2/health/ready` | None (internal network only) |
| `docker-compose.debug.yml` | Debug / host access | `/v2/health/live` | All services exposed to host |

**Debug compose** uses `/v2/health/live` instead of `/v2/health/ready` so Triton passes
health checks even when GPU models fail to load (e.g., no GPU available). This allows
triton-sdk and other dependent services to start.

## Services Expected

| Service | Image | Internal Ports | Debug Host Ports | Health Check |
|---------|-------|----------------|------------------|--------------|
| Redis | redis:7-alpine | 6379 | 0.0.0.0:6379 | `redis-cli ping` → PONG |
| Triton | nvcr.io/nvidia/tritonserver:${TRITON_VERSION:-25.03}-py3 | 8000 (HTTP), 8001 (gRPC), 8002 (metrics) | 0.0.0.0:8000-8002 | `curl http://localhost:8000/v2/health/live` → 200 |
| RedisInsight | redis/redisinsight:latest | 5540 | 0.0.0.0:5540 | Container running |
| Triton SDK | nvcr.io/nvidia/tritonserver:${TRITON_VERSION:-25.03}-py3-sdk | — | 0.0.0.0:8010 | Container running |

`TRITON_VERSION` comes from the shell environment or `.devcontainer/.env` (the Compose project directory is `.devcontainer/`); the repo-root `.env` is not read by these commands, so export the variable or pass `--env-file .env` to switch versions.

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
curl -sf http://triton:8000/v2/health/live && echo OK || echo FAIL
```
Expected: `OK`. If connection refused: Triton container is not running or still loading.

#### Triton Models
```bash
curl -s http://triton:8000/v2/repository/index -X POST -d {} | python3 -m json.tool
```
List all models in the repository with their state (READY vs UNAVAILABLE).

#### If ANY service is unreachable

**STOP immediately.** Do NOT attempt any workarounds. Tell the user:

> I'm running inside the devcontainer and cannot manage Docker services directly.
> The following service(s) are unreachable: [list them].
>
> Please run these commands from your **host terminal** (outside the container):
>
> ```bash
> # Check what's running (use debug compose for exposed ports)
> docker compose -f .devcontainer/docker-compose.debug.yml ps
>
> # Start all services with ports exposed
> docker compose -f .devcontainer/docker-compose.debug.yml up -d
>
> # Check logs for a failing service
> docker compose -f .devcontainer/docker-compose.debug.yml logs <service> --tail=50
> ```
>
> After fixing, ask me to re-run `/check-infra`.

**Do NOT proceed** with any other work that depends on infrastructure until the user confirms services are fixed.

After verifying services from inside the container, skip Steps 1-3 and go directly to Step 4 (model repository check) and Step 5 (summary).

---

## Step 1: Check Service Status

> **Only when running OUTSIDE the container (not vscode user).**

Determine which compose file is in use. Prefer debug compose if it exists:

```bash
docker compose -f .devcontainer/docker-compose.debug.yml ps
```

If no containers running with debug compose, try default:
```bash
docker compose -f .devcontainer/docker-compose.yml ps
```

Report which services are running, stopped, or missing.

## Step 2: Health Checks

> **Only when running OUTSIDE the container (not vscode user).**

For each running service, verify health:

### Redis
```bash
docker compose -f .devcontainer/docker-compose.debug.yml exec redis redis-cli ping
```
Expected: `PONG`. If it fails: Redis is starting or crashed.

### Triton HTTP Liveness
```bash
curl -s http://localhost:8000/v2/health/live && echo OK || echo FAIL
```
Expected: HTTP 200 + `OK`. Liveness means Triton is running.

### Triton HTTP Readiness
```bash
curl -s http://localhost:8000/v2/health/ready && echo READY || echo NOT_READY
```
Readiness means models are loaded. May be `NOT_READY` without GPU — this is expected.

### Triton Models
```bash
curl -s http://localhost:8000/v2/repository/index -X POST -d {} | python3 -m json.tool
```
List all models in the repository with their state (READY vs UNAVAILABLE).

### RedisInsight
```bash
curl -s http://localhost:5540 -o /dev/null -w %{http_code}
```
Expected: `200`.

### Triton Metrics
```bash
curl -s http://localhost:8002/metrics | head -5
```
Expected: Prometheus-format metrics.

## Step 3: Diagnose Failures

> **Only when running OUTSIDE the container (not vscode user).**

If Triton is unhealthy or models are UNAVAILABLE:

```bash
docker compose -f .devcontainer/docker-compose.debug.yml logs triton --tail=50
```

### Common Triton Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `model configuration not found` | Missing config.pbtxt | Create config.pbtxt (strict mode requires it) |
| `Internal: unable to load` | config.pbtxt parse error | Check syntax, field names, data types |
| `GPU not available` / `KIND_GPU but no GPUs` | No GPU or nvidia-docker issue | Expected in CPU-only environments. Models stay UNAVAILABLE but Triton is live. |
| `model version X not found` | Missing version directory | Create `1/` directory with model file |
| `--exit-on-error=false` but no models | Empty model_repository | Export models first with `uv run triton-model-tools convert` |
| triton-sdk didn't start | Triton unhealthy | Use debug compose (live health check) or start triton-sdk manually |

### Common Redis Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Connection refused | Container not running | `docker compose -f .devcontainer/docker-compose.debug.yml up -d redis` |
| `LOADING Redis is loading` | AOF replay in progress | Wait for loading to complete |

## Step 4: Check Model Repository

```bash
ls -la model_repository/
```

Compare local models vs what Triton reports as loaded. Each model needs:
- `config.pbtxt` (required with `--strict-model-config=true`)
- At least one version directory (`1/`) containing the model file

Both compose files run Triton with `--model-control-mode=explicit`; a model present in `model_repository/` stays unloaded until loaded via the Model Control API (see the `using-triton-inference-server` skill).

## Step 5: Summary

Output a status table:

```
| Service      | Status  | Details                          |
|-------------|---------|----------------------------------|
| Redis        | UP      | PONG response OK, port 6379      |
| Triton       | UP      | Live, 1 model READY, port 8000   |
| RedisInsight | UP      | Web UI at localhost:5540         |
| Triton SDK   | UP      | Container running, port 8010     |
```

If services are down and you are OUTSIDE the container, suggest:
```bash
docker compose -f .devcontainer/docker-compose.debug.yml up -d
```

If services are down and you are INSIDE the container (vscode), follow the
"If ANY service is unreachable" instructions from Step 0.
