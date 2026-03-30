# triton-redis-connector

Микросервисный компонент для инференса ML-моделей через NVIDIA Triton Inference Server с использованием Redis Streams в качестве транспорта.

## Пакеты

| Пакет | Описание |
|-------|----------|
| `triton-redis-connector` | Коннектор Redis Streams ↔ Triton Inference Server |
| `triton-model-tools` | Конвертация и верификация ultralytics-моделей для Triton |

## Быстрый старт

```bash
# Установка зависимостей
uv sync --dev

# Запуск инфраструктуры
docker compose up -d redis              # только Redis
docker compose up -d                    # Redis + Triton (требует GPU)

# Проверка
uv run pytest
uv run pre-commit run --all-files
```

## Разработка

```bash
uv run ruff format .                    # форматирование
uv run ruff check .                     # линтер
uv run mypy .                           # проверка типов
uv run pyright                          # проверка типов (pyright)
uv run vulture                          # поиск мёртвого кода
uv run pytest                           # юнит-тесты
uv run pytest -m integration -n 0      # интеграционные тесты (требует Docker)
```

## Инфраструктура

| Сервис | Порт | Описание |
|--------|------|----------|
| Redis | 6379 | Очередь сообщений (Streams) |
| Triton HTTP | 8000 | HTTP API инференса |
| Triton gRPC | 8001 | gRPC API инференса |
| Triton Metrics | 8002 | Prometheus-метрики |
| RedisInsight | 5540 | Веб-UI для Redis |
