# Внедрение зависимостей (Dishka)

typed-flow использует [dishka](https://dishka.readthedocs.io/) для управления зависимостями. Это опционально — можно работать без DI, создавая зависимости вручную.

## Зачем DI

- **Один Redis-клиент** на приложение вместо создания отдельного в каждом источнике/приёмнике
- **Тестируемость** — подмена зависимостей в тестах через `override`
- **Управление жизненным циклом** — контейнер создаёт и закрывает ресурсы

## TypedFlowProvider

Встроенный провайдер предоставляет базовые зависимости из `AppConfig`:

```python
from typed_flow.di import TypedFlowProvider, make_app_container
```

Что предоставляет:

| Зависимость | Scope | Описание |
|-------------|-------|----------|
| `RedisConnectionConfig` | APP | Извлекается из `AppConfig.redis` |
| `Redis[bytes]` | APP | Потокобезопасный Redis-клиент |

## Использование

### Базовый сценарий

```python
from redis import Redis
from typed_flow.config.schema import AppConfig, RedisConnectionConfig, SourceConfig, SinkConfig
from typed_flow.connectors import RedisThreadSource, RedisThreadSink
from typed_flow.di import make_app_container
from typed_flow.engine.app import Application

config = AppConfig(redis=RedisConnectionConfig(host="redis", port=6379))
container = make_app_container(config)

# Один клиент для всего приложения
client = container.get(Redis[bytes])

source = RedisThreadSource[MyEvent](redis=config.redis, source=SourceConfig(
    stream_name="input:stream",
    consumer_group="my-group",
))
sink = RedisThreadSink[MyEvent, MyResult](redis=config.redis, sink=SinkConfig(
    stream_name="output:stream",
))

# Передаём клиент в Application.build() для метрик
app = Application.build(config, [source, MyProcessor(), sink], redis_client=client)
app.run()
```

### Создание контейнера вручную

```python
from dishka import make_container
from typed_flow.di import TypedFlowProvider

container = make_container(TypedFlowProvider(), context={AppConfig: config})
```

## Расширение провайдера

Добавьте свой провайдер для дополнительных зависимостей:

```python
from dishka import Provider, Scope, provide
import tritonclient.grpc as grpcclient

class TritonProvider(Provider):
    scope = Scope.APP

    @provide
    def triton_client(self) -> grpcclient.InferenceServerClient:
        return grpcclient.InferenceServerClient(url="localhost:8001")

# Объединяем провайдеры
container = make_container(
    TypedFlowProvider(),
    TritonProvider(),
    context={AppConfig: config},
)
```

## Тестирование

Переопределите зависимости через `override`:

```python
from unittest.mock import Mock
from dishka import Provider, Scope, provide, make_container
from redis import Redis

class MockRedisProvider(Provider):
    @provide(scope=Scope.APP, override=True)
    def redis_client(self) -> Redis[bytes]:
        return Mock(spec=Redis)

test_container = make_container(
    TypedFlowProvider(),
    MockRedisProvider(),
    context={AppConfig: config},
)

mock_redis = test_container.get(Redis[bytes])
# mock_redis — это Mock, не настоящий клиент
```

## Архитектура

```
AppConfig (context)
    ↓
TypedFlowProvider
    ├── RedisConnectionConfig (из config.redis)
    └── Redis[bytes] (один на приложение)
```

`di/` находится на том же уровне, что и `connectors/` в иерархии import-linter. Модуль `engine` не импортирует `di` — контейнер используется в пользовательском коде (main.py), не внутри фреймворка.
