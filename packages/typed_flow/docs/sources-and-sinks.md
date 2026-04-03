# Источники и приёмники

## Redis-источники

typed-flow предоставляет два варианта Redis-источников с **идентичным API**:

| Класс | Вариант исполнения | Создание Redis-клиента |
|-------|--------------------|----------------------|
| `RedisThreadSource[T]` | Выделенный поток | Немедленно в `__init__` |
| `RedisProcessSource[T]` | Отдельный OS-процесс | Лениво при первом `__call__` |

### Использование

```python
from typed_flow.config.schema import RedisConnectionConfig, SourceConfig
from typed_flow.connectors import RedisThreadSource, RedisProcessSource

redis_cfg = RedisConnectionConfig(host="localhost", port=6379)
source_cfg = SourceConfig(
    stream_name="camera:frames",
    consumer_group="detector-group",
    event_model="FrameEvent",
    batch_size=1,
    block_ms=100,
)

# Потоковый вариант
source = RedisThreadSource[FrameEvent](redis=redis_cfg, source=source_cfg)

# Процессный вариант — тот же API, другой класс
source = RedisProcessSource[FrameEvent](redis=redis_cfg, source=source_cfg)
```

### Подклассы

```python
class CameraSource(RedisThreadSource[FrameEvent]):
    pass

source = CameraSource(redis=redis_cfg, source=source_cfg)
```

Модель события `T` извлекается из generic-параметра при определении класса через `__init_subclass__`.

### Как работает внутри

1. Источник создаёт `RedisStreamInputConnector` (обёртка над `XREADGROUP`)
2. Каждый вызов `__call__` читает одно сообщение из Redis Stream
3. Десериализует в `PipelineEvent[T, Unprocessed]` через Pydantic
4. При ошибке валидации возвращает `ConnectorError` (не выбрасывает исключение)

### Thread vs Process

- **RedisThreadSource** — создаёт Redis-клиент сразу в `__init__`. Клиент `redis.Redis` потокобезопасен, поэтому один клиент может быть разделён между потоками.
- **RedisProcessSource** — сохраняет конфигурацию, создаёт клиент при первом вызове `__call__` внутри дочернего процесса. Redis-клиент нельзя передавать между процессами.

---

## Redis-приёмники

Аналогичная пара с идентичным API:

| Класс | Вариант исполнения |
|-------|--------------------|
| `RedisThreadSink[TCtx, TResult]` | Выделенный поток |
| `RedisProcessSink[TCtx, TResult]` | Отдельный OS-процесс |

### Использование

```python
from typed_flow.config.schema import SinkConfig
from typed_flow.connectors import RedisThreadSink, RedisProcessSink

sink_cfg = SinkConfig(
    stream_name="detections:output",
    maxlen=100_000,
)

sink = RedisThreadSink[FrameEvent, DetectionResult](redis=redis_cfg, sink=sink_cfg)

# Или процессный вариант
sink = RedisProcessSink[FrameEvent, DetectionResult](redis=redis_cfg, sink=sink_cfg)
```

### Сериализация

Приёмник записывает в Redis Stream:
- Все поля `context` (`TCtx`) как отдельные ключи хеша
- `payload` (`TResult`) как JSON в поле `result`
- Метаданные фреймворка (`fw`) **не записываются**

---

## Кастомные источники

Для нестандартных источников данных наследуйтесь от базовых классов:

```python
from typed_flow.protocols import BaseThreadSource
from typed_flow.engine.events import PipelineEvent, Unprocessed

class FileSource(BaseThreadSource[PipelineEvent[MyEvent, Unprocessed]]):
    def __init__(self, path: str) -> None:
        super().__init__()
        self._lines = open(path).readlines()
        self._index = 0
        self.name = "file_source"

    def __call__(self) -> PipelineEvent[MyEvent, Unprocessed] | None:
        if self._index >= len(self._lines):
            return None
        line = self._lines[self._index]
        self._index += 1
        event = MyEvent.model_validate_json(line)
        return PipelineEvent(context=event, payload=Unprocessed())
```

Требования:
- Наследование от `BaseThreadSource[TOut]` или `BaseProcessSource[TOut]`
- Метод `__call__() -> TOut | None` — возвращает событие или `None`
- Атрибут `stop: threading.Event` (устанавливается фреймворком при завершении)

## Кастомные приёмники

```python
from typed_flow.protocols import BaseThreadSink

class LogSink(BaseThreadSink[PipelineEvent[MyEvent, MyResult]]):
    def __call__(self, event: PipelineEvent[MyEvent, MyResult]) -> None:
        logger.info("Result: {}", event.payload)
```

---

## Легаси классы

Следующие классы из `typed_flow.connectors.redis_stream` по-прежнему доступны:

- `AutoRedisSource[T]` — требует ручного создания `RedisStreamInputConnector`
- `LiveRedisStreamSource[T]` — принимает явный `event_model` параметр
- `TypedRedisStreamSink[TCtx, TResult]` — требует ручного создания `RedisStreamOutputConnector`

Рекомендуется использовать `RedisThreadSource` / `RedisThreadSink` вместо них.
