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

## Метрики

Все метрики отправляются в Redis Stream (по умолчанию `metrics`) в формате `MetricEvent` с полями `component_name`, `measure`, `value`, `timestamp_ms`, `tags`. При отсутствии входных событий все метрики отправляются со значением `0`.

Интервал отправки настраивается через `metrics.stats_log_interval_s` (по умолчанию 60 с).

### Pipeline метрики

| Метрика | Компонент | Теги | Тип | Описание |
|---------|-----------|------|-----|----------|
| `stats_processed` | app_name | — | counter | Кумулятивное число обработанных событий (дошедших до Sink) |
| `stats_in_progress` | app_name | — | gauge | Число событий в обработке прямо сейчас |
| `stats_errors` | app_name | — | counter | Кумулятивное число ошибок |
| `event_total_transit_ms` | app_name | — | histogram | Время от создания Envelope до записи в Sink (мс) |
| `inference_time_ms` | app_name | — | histogram | Время инференса из DetectionResult (мс) |
| `parallel_restart` | app_name | `node` | counter | Перезапуск параллельного потока |

### Per-node метрики

| Метрика | Компонент | Теги | Тип | Описание |
|---------|-----------|------|-----|----------|
| `node_events_total` | app_name | `node` | counter | Кумулятивное число вызовов узла |
| `node_events_interval` | app_name | `node` | gauge | Число событий за последний интервал статистики |
| `node_processing_ms` | app_name | `node` | histogram | Время обработки одного события узлом (мс) |
| `node_errors` | app_name | `node` | counter | Кумулятивное число ошибок по узлам |
| `source_read_total` | app_name | `node` | counter | Общее число попыток чтения источником |
| `source_read_empty` | app_name | `node` | counter | Попытки чтения, вернувшие None |
| `sink_write_total` | app_name | `node` | counter | Общее число записей в Sink |
| `decode_success` | app_name | — | counter | Успешные декодирования изображений |
| `decode_failure` | app_name | — | counter | Ошибки декодирования изображений |

### Queue метрики

| Метрика | Компонент | Теги | Тип | Описание |
|---------|-----------|------|-----|----------|
| `queue_depth` | app_name | `queue` | gauge | Текущая глубина очереди между узлами |
| `queue_drops_total` | app_name | `queue` | counter | Кумулятивное число сброшенных сообщений (backpressure) |

### Triton метрики (gRPC stats polling)

| Метрика | Компонент | Теги | Тип | Описание |
|---------|-----------|------|-----|----------|
| `inference_count` | triton | `model` | counter | Число инференсов |
| `execution_count` | triton | `model` | counter | Число выполнений |
| `success_count` | triton | `model` | counter | Число успешных инференсов |
| `success_time_ms` | triton | `model` | counter | Суммарное время успешных инференсов (мс) |
| `fail_count` | triton | `model` | counter | Число неудачных инференсов |
| `queue_time_ms` | triton | `model` | counter | Время в очереди Triton (мс) |
| `compute_infer_time_ms` | triton | `model` | counter | Время вычислений Triton (мс) |

### Annotated-based инструментация

Для параллельных узлов (multi-thread / multi-process) используйте `Annotated` с маркером `Tracked` для автоматического сбора метрик через `MetricAggregator`:

```python
from typing import Annotated
from triton_redis_connector.engine.metric_aggregator import Tracked

InferenceType = Annotated[InferenceWorker, Tracked("inference")]
```

`MetricAggregator` автоматически собирает `node_events_total`, `node_events_interval`, `node_processing_ms`, `node_errors`, `source_read_empty` для каждого помеченного узла.

## Примеры

- [`examples/camera_inference/`](examples/camera_inference/) -- минимальный демо-пайплайн с синтетическими данными
- [`examples/yolo_detection_pipeline/`](examples/yolo_detection_pipeline/) -- полный пайплайн YOLO-детекции через Redis + Triton
