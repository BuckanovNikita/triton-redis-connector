# Примеры / Examples

## Можно ли запустить приложение только через YAML?

**Частично.** Вся инфраструктурная конфигурация (Redis, Triton, очереди, метрики) задаётся
в `config.yaml` и поддерживает переменные окружения через `${oc.env:VAR,default}`.

Однако **процессоры требуют Python-кода**, потому что:
- Каждый процессор — это класс с типизированным `__call__` 
- Фреймворк строит DAG автоматически по аннотациям типов
- Нет встроенного реестра процессоров по имени (поле `processors` в конфиге зарезервировано, но не реализовано)

### Что настраивается через YAML

| Секция | Что настраивает |
|---|---|
| `redis` | Хост, порт, номер БД |
| `sources` | Входные Redis Streams (имя, consumer group, batch size) |
| `sinks` | Выходные Redis Streams (имя, maxlen) |
| `inference` | Triton gRPC URL, число воркеров, retry-политика |
| `preprocessing` | Декодер (opencv/turbojpeg), размер тензора |
| `queues` | Размер внутренних очередей |
| `metrics` | Стрим метрик, интервалы опроса |

### Что требует Python-кода

- Реализация процессоров (классы с `__call__`)
- Регистрация узлов в `PipelineRegistry`
- Создание `Application` и вызов `app.run()`

### Минимальный шаблон запуска

```python
from pathlib import Path
from triton_redis_connector.config.hydra_app import load_config
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry

config = load_config(config_name="config", config_dir=Path(__file__).parent)

registry = PipelineRegistry()
registry.register("source", MySource())
registry.register("processor", MyProcessor())
registry.register("sink", MySink())

app = Application(config=config, registry=registry, queue_bus=QueueBus(config=config.queues))
app.run()
```

Все параметры инфраструктуры берутся из `config.yaml`, а переопределить их можно
переменными окружения:

```bash
REDIS_HOST=localhost TRITON_GRPC_URL=localhost:8001 uv run python my_pipeline.py
```

---

## Примеры

### 1. YOLO Detection Pipeline (полный)

Реальный пайплайн: Redis Stream → декодирование JPEG → letterbox → Triton → NMS → Redis Stream.

```
examples/yolo_detection_pipeline/
├── config.yaml        — конфигурация (Redis, Triton, очереди)
├── run_pipeline.py    — точка входа (загрузка конфига + регистрация узлов)
├── processors.py      — процессоры (декодер, препроцессор, постпроцессор, сериализатор)
└── stream_images.py   — утилита: загрузить изображения в Redis Stream (batch или streaming)
```

**Запуск:**

```bash
# 1. Загрузить тестовые изображения в Redis (batch — один раз и выход)
uv run python examples/yolo_detection_pipeline/stream_images.py

# 2. Запустить пайплайн
uv run python examples/yolo_detection_pipeline/run_pipeline.py

# Или: непрерывный стриминг с заданным FPS
uv run python examples/yolo_detection_pipeline/stream_images.py --fps 10
```

**Граф:**

```
RedisStreamSource → EnvelopeExtractor → ImageDeserializer
  → PreprocessorNode → InferenceWorker → DetectionPostProcessor
  → OutputSerializer → RedisStreamSink
```

### 2. Camera Inference (демо)

Минимальный пример с синтетическими данными — без реального инференса.
Показывает как фреймворк автоматически строит DAG по типовым аннотациям.

```
examples/camera_inference/
├── config.yaml       — конфигурация
├── run_pipeline.py   — точка входа
├── processors.py     — синтетический Source, Processor, Sink
└── events.py         — модели событий (CameraFrame, DetectionResult)
```

**Запуск:**

```bash
uv run python examples/camera_inference/run_pipeline.py
```
