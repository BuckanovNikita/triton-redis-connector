# Примеры / Examples

## Можно ли запустить приложение только через YAML?

**Частично.** Вся инфраструктурная конфигурация (Redis, Triton, очереди, метрики) задаётся
в `config.yaml` и поддерживает переменные окружения через `${oc.env:VAR,default}`.

Однако **процессоры требуют Python-кода**, потому что:
- Каждый процессор -- это класс с типизированным `__call__` 
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

### Dual Model Pipeline (полный)

Пайплайн стальной инспекции с двумя моделями: DALI-ensemble (чётные камеры) и TurboJPEG + yolo26l (нечётные камеры). Использует shipped-процессоры: `EnvelopeExtractor`, `EventSerializer`, `LogErrorSink`.

```
examples/dual_model_pipeline/
├── config.yaml           — конфигурация (Redis, Triton, очереди)
├── run_pipeline.py       — точка входа (загрузка конфига + регистрация узлов)
├── processors.py         — доменные процессоры (роутинг, препроцессинг, инференс, постпроцессинг)
├── events.py             — модели событий с Counter/Gauge/GroupBy метриками
└── test_deterministic.py — детерминистический тест без Triton
```

**Запуск:**

```bash
# 1. Заполнить Redis Stream тестовыми данными
uv run triton-model-tools populate-stream --image-dir ./images --stream steel:inspection_input

# 2. Запустить пайплайн
uv run python examples/dual_model_pipeline/run_pipeline.py
```

**Граф:**

```
RedisStreamSource -> EnvelopeExtractor -> CameraBranchRouter
  -> [even] DaliPreprocessor --------+
  -> [odd]  TurboJpegPreprocessor ---+-> MetadataInferenceWorker
                                         -> InspectionPostProcessor
                                             -> EventSerializer -> RedisStreamSink
                                         -> LogErrorSink (InferenceError)
```
