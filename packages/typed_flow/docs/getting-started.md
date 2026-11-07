# Начало работы с typed-flow

## Что такое typed-flow

typed-flow — фреймворк потоковой обработки событий с маршрутизацией по типам. Пайплайн строится из трёх видов узлов:

- **Source** — источник событий (Redis Stream, очередь, файл)
- **Processor** — обработчик: принимает событие одного типа, возвращает другого
- **Sink** — приёмник: записывает результат (Redis Stream, лог, БД)

Фреймворк автоматически строит граф обработки на основе типовых аннотаций и маршрутизирует события с поддержкой подтипов (`isinstance`), union-входов и распаковки `list[T]`/`tuple[T, ...]`.

## Установка

```bash
uv add typed-flow
```

Зависимости: `redis`, `pydantic`, `loguru`, `dishka`.

## Ключевые концепции

### PipelineEvent

Каждое событие в пайплайне обёрнуто в `PipelineEvent[TCtx, TPayload]`:

```python
from typed_flow.engine.events import PipelineEvent, Unprocessed

# Источник создаёт PipelineEvent[MyEvent, Unprocessed]
# Процессор преобразует в PipelineEvent[MyEvent, MyResult]
# Приёмник получает PipelineEvent[MyEvent, MyResult]
```

- `context` (`TCtx`) — исходное пользовательское событие, не меняется
- `payload` (`TPayload`) — результат обработки, меняется на каждом этапе
- `fw` — метаданные фреймворка (stream, message_id), не сериализуются в выход

### Маршрутизация по типам

Фреймворк извлекает типы из аннотаций `__call__` каждого узла и строит DAG:

```python
class MyProcessor(
    BaseThreadProcessor[
        PipelineEvent[MyEvent, Unprocessed],  # вход
        PipelineEvent[MyEvent, DetectionResult],  # выход
    ]
):
    def __call__(self, event):
        # event имеет тип PipelineEvent[MyEvent, Unprocessed]
        result = DetectionResult(...)
        return PipelineEvent(fw=event.fw, context=event.context, payload=result)
```

Если процессор возвращает `Union` типов — фреймворк разветвляет поток к разным потребителям.

Маршрутизация поддерживает:

- **Подтипы (наследование)**: `ChildEvent` маршрутизируется к потребителю `ParentEvent` через `isinstance`
- **Union-входы**: `BaseThreadProcessor[A | B | C, Out]` подписывается на все три типа
- **Контейнеры**: `list[T]` и `tuple[T, ...]` распаковываются — каждый элемент маршрутизируется отдельно

## Минимальный пример

```python
from pydantic import BaseModel
from typed_flow.config.schema import (
    AppConfig,
    RedisConnectionConfig,
    SourceConfig,
    SinkConfig,
)
from typed_flow.connectors import RedisThreadSource, RedisThreadSink
from typed_flow.engine.app import Application
from typed_flow.engine.events import PipelineEvent, Unprocessed
from typed_flow.protocols import BaseThreadProcessor


# 1. Модели событий
class InputEvent(BaseModel):
    camera_id: str
    frame_id: int


class OutputResult(BaseModel):
    camera_id: str
    frame_id: int
    status: str


# 2. Источник — две строки
class MySource(RedisThreadSource[InputEvent]):
    pass


# 3. Процессор
class MyProcessor(
    BaseThreadProcessor[
        PipelineEvent[InputEvent, Unprocessed],
        PipelineEvent[InputEvent, OutputResult],
    ]
):
    def __call__(self, event):
        result = OutputResult(
            camera_id=event.context.camera_id,
            frame_id=event.context.frame_id,
            status="processed",
        )
        return PipelineEvent(fw=event.fw, context=event.context, payload=result)


# 4. Сборка и запуск
redis_cfg = RedisConnectionConfig(host="localhost", port=6379)

source = MySource(
    redis=redis_cfg,
    source=SourceConfig(stream_name="input:stream", consumer_group="my-group"),
)
sink = RedisThreadSink[InputEvent, OutputResult](
    redis=redis_cfg,
    sink=SinkConfig(stream_name="output:stream"),
)

app = Application.build(
    AppConfig(redis=redis_cfg),
    processors=[source, MyProcessor(), sink],
)
app.run()
```

## Конфигурация

### RedisConnectionConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `host` | `str` | `"localhost"` | Хост Redis |
| `port` | `int` | `6379` | Порт Redis |
| `db` | `int` | `0` | Номер БД |

### SourceConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `stream_name` | `str` | — | Имя входного потока |
| `consumer_group` | `str` | — | Имя consumer group |
| `consumer_name` | `str` | `""` | Имя consumer (авто) |
| `event_model` | `str` | — | Имя модели события |
| `batch_size` | `int` | `10` | Размер батча |
| `block_ms` | `int` | `0` | Таймаут блокировки (мс) |

### SinkConfig

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `stream_name` | `str` | — | Имя выходного потока |
| `maxlen` | `int \| None` | `10_000` | Макс. длина потока |
| `approximate_maxlen` | `bool` | `True` | Приблизительная обрезка |

### AppConfig

Объединяет все конфигурации: `redis`, `sources`, `sinks`, `queues`, `metrics`, `debug`.

## Следующие шаги

- [Источники и приёмники](sources-and-sinks.md) — подробно о `RedisThreadSource`, `RedisProcessSource`, кастомных источниках
- [Внедрение зависимостей](dependency-injection.md) — dishka DI для управления Redis-клиентами
- [Варианты исполнения](execution-variants.md) — потоки vs процессы, пулы
