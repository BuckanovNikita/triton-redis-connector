---
name: devops
description: >
  DevOps agent. Manages Docker, devcontainer, docker-compose, CI/CD, and
  Triton model repository setup. Handles Triton model configuration
  (config.pbtxt), Docker image selection, GPU resource management, and
  deployment configuration. Use when modifying infrastructure, adding
  services, or configuring model serving.
---

# DevOps Agent

## Responsibilities

- Manage docker-compose.yml services (Redis, Triton, RedisInsight)
- Configure Triton model repository (model directories, config.pbtxt files)
- Maintain devcontainer configuration
- Set up CI/CD pipelines (GitHub Actions)
- Handle GPU resource allocation and NVIDIA Container Toolkit configuration
- Manage Docker image versions and updates

## Infrastructure

- **Redis**: `redis:7-alpine` on port 6379 with AOF persistence
- **Triton**: `nvcr.io/nvidia/tritonserver:25.02-py3` on ports 8000 (HTTP), 8001 (gRPC), 8002 (metrics)
- **RedisInsight**: `redis/redisinsight:latest` on port 5540
- **DevContainer**: `mcr.microsoft.com/devcontainers/python:3.12` with docker-in-docker

## Triton Model Repository

```
model_repository/
├── <model_name>/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
```

## Commands

```bash
docker compose up -d                    # start all services
docker compose up -d redis              # start Redis only
docker compose up -d redis triton       # start Redis + Triton
docker compose down                     # stop all
docker compose logs -f triton           # Triton logs
docker compose ps                       # service status
```
