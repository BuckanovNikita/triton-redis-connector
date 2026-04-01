# triton-redis-connector

Микросервисный компонент для инференса ML-моделей через NVIDIA Triton Inference Server с использованием Redis Streams в качестве транспорта.

## Архитектура

```mermaid
graph LR
    subgraph Источники данных
        CAM[Камеры / Сенсоры]
        FILES[Файлы / API]
    end

    subgraph Redis
        INPUT_STREAM[(Redis Streams<br/>входные)]
        OUTPUT_STREAM[(Redis Streams<br/>результаты)]
        METRICS_STREAM[(Redis Streams<br/>метрики)]
    end

    subgraph triton-redis-connector
        SRC([Source]) --> PROC[Processor]
        PROC --> INF[Inference Worker]
        INF --> SINK[[Sink]]
    end

    subgraph Triton Server
        TRITON[NVIDIA Triton<br/>Inference Server]
    end

    CAM --> INPUT_STREAM
    FILES --> INPUT_STREAM
    INPUT_STREAM --> SRC
    INF -->|gRPC| TRITON
    SINK --> OUTPUT_STREAM
    SINK --> METRICS_STREAM
```

## Пакеты

| Пакет | Путь | Описание |
|-------|------|----------|
| `triton-redis-connector` | `packages/triton_redis_connector/` | Коннектор Redis Streams <-> Triton: граф-пайплайн с 9 вариантами исполнения |
| `triton-model-tools` | `packages/triton_model_tools/` | CLI для конвертации (`convert`), экспорта, бенчмаркинга и верификации YOLO-моделей для Triton |
| `triton-shared-models` | `packages/triton_shared_models/` | Общие типы Triton (`TritonModelConfig`, `TritonDataType`, `config.pbtxt`) и letterbox-препроцессинг |

Полная архитектура описана в [docs/design.md](docs/design.md).

## Быстрый старт

```bash
# Установка зависимостей
uv sync --dev

# Запуск инфраструктуры
docker compose up -d redis              # только Redis
docker compose up -d                    # Redis + Triton (требует GPU)

# Проверка
uv run pre-commit run --all-files
uv run pytest
```

## Разработка

| Команда | Описание |
|---------|----------|
| `uv run ruff format .` | Форматирование кода |
| `uv run ruff check .` | Линтер (ruff) |
| `uv run ruff check --fix .` | Автоисправление линтера |
| `uv run mypy .` | Проверка типов (mypy strict) |
| `uv run pyright` | Проверка типов (pyright strict) |
| `uv run lint-imports` | Проверка архитектурных контрактов импортов |
| `uv run vulture` | Поиск мертвого кода |
| `uv run pytest` | Запуск тестов (параллельно) |
| `uv run pytest -x` | Остановка на первом падении |
| `uv run pytest -k test_name` | Запуск конкретного теста |
| `uv run pre-commit run --all-files` | Все проверки по порядку |

## Инфраструктура (Docker Compose)

Инфраструктура определена в `.devcontainer/docker-compose.yml`:

| Сервис | Образ | Порт | Описание |
|--------|-------|------|----------|
| Redis | `redis:7-alpine` | 6379 | Очередь сообщений (Streams), AOF persistence |
| Triton | `nvcr.io/nvidia/tritonserver:26.02-py3` | 8000 (HTTP), 8001 (gRPC), 8002 (Metrics) | Inference Server, `--strict-model-config=true` |
| Triton SDK | `nvcr.io/nvidia/tritonserver:26.02-py3-sdk` | -- | `perf_analyzer`, `model_analyzer`, `tritonclient` |
| RedisInsight | `redis/redisinsight:latest` | 5540 | Веб-UI для Redis |

## Примеры

- [`examples/camera_inference/`](examples/camera_inference/) -- минимальный демо-пайплайн с синтетическими данными
- [`examples/yolo_detection_pipeline/`](examples/yolo_detection_pipeline/) -- полный пайплайн YOLO-детекции через Redis + Triton
