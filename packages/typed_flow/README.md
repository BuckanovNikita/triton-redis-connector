# typed-flow

Фреймворк потоковой обработки событий с маршрутизацией по типам.

## Возможности

- **Типизированный DAG** — граф обработки строится автоматически из аннотаций типов
- **Redis Streams** — встроенные источники и приёмники для Redis
- **Варианты исполнения** — потоки, процессы, пулы потоков/процессов
- **Dishka DI** — внедрение зависимостей для управления Redis-клиентами
- **Метрики** — Counter, Gauge, GroupBy на event-классах

## Быстрый старт

```python
from typed_flow.config.schema import AppConfig, RedisConnectionConfig, SourceConfig, SinkConfig
from typed_flow.connectors import RedisThreadSource, RedisThreadSink
from typed_flow.engine.app import Application

redis_cfg = RedisConnectionConfig(host="localhost", port=6379)

source = RedisThreadSource[MyEvent](
    redis=redis_cfg,
    source=SourceConfig(stream_name="input", consumer_group="grp"),
)
sink = RedisThreadSink[MyEvent, MyResult](
    redis=redis_cfg,
    sink=SinkConfig(stream_name="output"),
)

app = Application.build(AppConfig(redis=redis_cfg), [source, processor, sink])
app.run()
```

## Документация

- [Начало работы](docs/getting-started.md)
- [Источники и приёмники](docs/sources-and-sinks.md)
- [Внедрение зависимостей](docs/dependency-injection.md)
- [Варианты исполнения](docs/execution-variants.md)
