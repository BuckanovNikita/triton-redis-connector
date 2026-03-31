# Пример: Camera Inference (демо)

Минимальный пример пайплайна с синтетическими данными — без реального инференса.
Демонстрирует автоматическую маршрутизацию событий по типовым аннотациям `__call__`.

## Структура

```
config.yaml      — конфигурация (Hydra + переменные окружения)
events.py        — модели событий: CameraFrame, DetectionResult
processors.py    — Source, Processor, Sink (синтетические данные)
run_pipeline.py  — точка входа
```

## Граф пайплайна

```
CameraSource (ThreadSource)  — генерирует фейковые кадры
    │
    ▼
FramePreprocessor (Processor) — создаёт синтетические детекции
    │
    ▼
ResultSink (Sink)             — логирует результаты
```

## Запуск

```bash
uv run python examples/camera_inference/run_pipeline.py
```

Ctrl+C для остановки.

## Конфигурация

Настройки в `config.yaml`. Переопределение через переменные окружения:

```bash
REDIS_HOST=localhost REDIS_PORT=6379 uv run python examples/camera_inference/run_pipeline.py
```
