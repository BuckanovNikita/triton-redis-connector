# Примеры / Reference Implementation

## Архитектура событий

Пользовательские модели событий -- обычные `BaseModel` (pydantic), без наследования от фреймворка.
Фреймворк оборачивает их в `PipelineEvent[TCtx, TPayload]`, где:
- `fw: FrameworkMeta` -- метаданные фреймворка (source_stream, message_id)
- `context: TCtx` -- пользовательский контекст (чистый BaseModel)
- `payload: TPayload` -- текущий этап обработки

Источники возвращают `PipelineEvent[T, Unprocessed]`. Роутеры преобразуют контекст и назначают payload.
Стоки (`TypedRedisStreamSink`) сериализуют контекст + payload в Redis автоматически.

## Что настраивается через YAML

| Секция | Что настраивает |
|---|---|
| `redis` | Хост, порт, номер БД |
| `sources` | Входные Redis Streams (имя, consumer group, batch size) |
| `sinks` | Выходные Redis Streams (имя, maxlen) |
| `inference` | Triton gRPC URL, число воркеров, retry-политика |
| `preprocessing` | Декодер (opencv/turbojpeg), размер тензора |
| `pipeline` | Модель Triton, имена тензоров |
| `queues` | Размер внутренних очередей |
| `metrics` | Стрим метрик, интервалы опроса |

## Что требует Python-кода

- Пользовательские pydantic-модели событий
- Роутеры (разветвление по типу задачи, номеру камеры и т.д.)
- Постпроцессор (`YoloPostProcessorBridge` или кастомный)

## Примеры

### Multi-task Pipeline (detection + classification)

Два типа задач: детекция (YOLO) и классификация (ResNet18). Каждая ветка имеет свой
`ClientPreprocessorNode`, `InferenceNode`, `PostProcessorNode` и `TypedRedisStreamSink`.

```
reference_implementation/
├── config.yaml       — конфигурация (Redis, Triton, очереди)
├── app.py            — multi-task pipeline (detection + classification)
├── run_pipeline.py   — dual-model pipeline (steel inspection)
├── processors.py     — роутеры (TaskRouter, CameraBranchRouter, ModelResultRouter)
├── events.py         — модели событий (чистые BaseModel)
```

**Граф multi-task:**

```
ImageTaskSource -> PipelineEvent[ImageTaskInput, Unprocessed]
  -> TaskRouter
    -> [detect]   ClientPreprocessorNode -> InferenceNode -> PostProcessorNode
                    -> TypedRedisStreamSink -> task:detection_results
    -> [classify] ClientPreprocessorNode -> InferenceNode -> PostProcessorNode
                    -> TypedRedisStreamSink -> task:classification_results
    -> LogErrorSinkNode
```

**Граф dual-model:**

```
SteelInspectionSource -> PipelineEvent[SteelInspectionInput, Unprocessed]
  -> CameraBranchRouter
    -> [even] ClientPreprocessorNode[SteelMeta, EvenImagePayload]
    -> [odd]  ClientPreprocessorNode[SteelMeta, OddImagePayload]
          -> InferenceNode (shared) -> PostProcessorNode (shared)
              -> ModelResultRouter
                -> TypedRedisStreamSink[SteelMeta, EvenDetectionResult]
                -> TypedRedisStreamSink[SteelMeta, OddDetectionResult]
              -> LogErrorSinkNode
```
