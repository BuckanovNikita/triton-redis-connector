# Пример: Camera Inference Pipeline

Пример приложения, демонстрирующего использование `triton-redis-connector` для построения
пайплайна обработки видеопотока с камер через Triton Inference Server.

## Структура

- `events.py` — доменные модели событий (`CameraFrame`, `DetectionResult`)
- `processors.py` — реализации Source, Processor и Sink для обработки кадров
- `config.yaml` — конфигурация Hydra для пайплайна

## Компоненты пайплайна

### CameraSource (ThreadSource)

Источник кадров камеры. В продакшн-версии читает из Redis Stream или API камеры.
В демонстрационном режиме генерирует синтетические кадры с заданным интервалом.

### FramePreprocessor (Processor)

Inline-процессор, преобразующий `CameraFrame` в `DetectionResult`.
В реальном приложении выполняет декодирование изображения, letterbox-препроцессинг,
вызов Triton для инференса и парсинг результатов.

### ResultSink (Sink)

Inline-приёмник, логирующий результаты детекции.
В продакшн-версии записывает результаты в Redis Stream или внешний сервис.

## Запуск

### Предварительные требования

- Запущенные Docker-сервисы Redis и Triton (через devcontainer)
- Установленные зависимости: `uv sync`

### Конфигурация

Настройки пайплайна задаются через `config.yaml` или переменные окружения:

- `REDIS_HOST` — хост Redis (по умолчанию: `localhost`)
- `REDIS_PORT` — порт Redis (по умолчанию: `6379`)
- `TRITON_URL` — URL Triton gRPC (по умолчанию: `triton:8001`)

### Переопределение параметров

Hydra поддерживает переопределение параметров через командную строку:

```bash
uv run python -m triton_redis_connector.main \
    redis.host=redis \
    inference.grpc_url=triton:8001 \
    app_name=my-camera-pipeline
```

## Архитектура

```
CameraSource (ThreadSource)
    │
    ▼
FramePreprocessor (Processor)
    │
    ▼
ResultSink (Sink)
```

Каждый компонент реализует один из 9 вариантов исполнения фреймворка.
Framework автоматически строит граф на основе типовых аннотаций `__call__`
и маршрутизирует события между компонентами.
