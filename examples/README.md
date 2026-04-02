# Примеры / Examples

## Можно ли запустить приложение только через YAML?

**Частично.** Вся инфраструктурная конфигурация (Redis, Triton, очереди, метрики) задаётся
в `config.yaml` и поддерживает переменные окружения через `${oc.env:VAR,default}`.

Однако **доменные процессоры требуют Python-кода**, потому что:
- Каждый процессор -- это класс с типизированным `__call__`
- Фреймворк строит DAG автоматически по аннотациям типов
- Shipped-процессоры (EnvelopeExtractor, DaliTritonRequestBuilder/ClientPreprocessor, InferenceWorker, EventSerializer, LogErrorSink) регистрируются автоматически через `build_application()` на основе полей конфигурации

### Что настраивается через YAML

| Секция | Что настраивает |
|---|---|
| `redis` | Хост, порт, номер БД |
| `sources` | Входные Redis Streams (имя, consumer group, batch size) |
| `sinks` | Выходные Redis Streams (имя, maxlen) |
| `inference` | Triton gRPC URL, число воркеров, retry-политика |
| `preprocessing` | Декодер (opencv/turbojpeg/dali), размер тензора |
| `pipeline` | Модель Triton, имена тензоров, пороги постпроцессора |
| `queues` | Размер внутренних очередей |
| `metrics` | Стрим метрик, интервалы опроса |

### Что требует Python-кода

- Класс события для десериализации (`event_model`)
- Постпроцессор (`YoloPostProcessorBridge` или кастомный)
- Пользовательские процессоры (`extra_nodes`)

### Минимальный шаблон запуска

```python
from pathlib import Path
from triton_redis_connector.config.hydra_app import load_config
from triton_redis_connector.providers.app_factory import build_application

config = load_config(config_name="config", config_dir=Path(__file__).parent)

app = build_application(
    config,
    event_model=MyInputEvent,
    postprocessor=MyPostProcessor(),
    extra_nodes=[
        ("my_processor", MyProcessor()),
    ],
)
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
├── processors.py         — один доменный процессор (CameraBranchRouter -- роутинг)
├── events.py             — модели событий (SteelInspectionInput, EvenCameraFrame, OddCameraFrame)
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
  -> [even] DaliTritonRequestBuilder ---+
  -> [odd]  ClientPreprocessor ----------+-> InferenceWorker
                                              -> YoloPostProcessorBridge
                                                  -> EventSerializer -> RedisStreamSink
                                              -> LogErrorSink (InferenceError)
```
