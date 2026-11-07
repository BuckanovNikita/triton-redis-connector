# Варианты исполнения

typed-flow поддерживает несколько вариантов параллельного исполнения узлов пайплайна. Выбор варианта определяет, как узел выполняется и какие ресурсы ему доступны.

## Обзор

| Вариант | Базовый класс | Параллелизм | Redis-клиент |
|---------|---------------|-------------|--------------|
| Thread Source | `BaseThreadSource[T]` | Выделенный поток | Общий (потокобезопасный) |
| Thread Processor | `BaseThreadProcessor[TIn, TOut]` | Выделенный поток | — |
| Thread Pool Processor | `BaseThreadPoolProcessor[TIn, TOut]` | N потоков, общая очередь | — |
| Thread Sink | `BaseThreadSink[T]` | Выделенный поток | Общий (потокобезопасный) |
| Process Source | `BaseProcessSource[T]` | Отдельный OS-процесс | Свой (лениво) |
| Process Pool Processor | `BaseProcessPoolProcessor[TIn, TOut]` | N процессов | — |
| Process Sink | `BaseProcessSink[T]` | Отдельный OS-процесс | Свой (лениво) |

## Потоковые варианты (Thread)

Используют `threading.Thread`. Все потоки разделяют память процесса.

**Когда использовать:**
- I/O-bound операции (Redis, HTTP, файловый ввод/вывод)
- Работа с GIL-совместимыми библиотеками
- Большинство задач — это вариант по умолчанию

**Redis-клиент:** `redis.Redis` потокобезопасен. Один клиент может использоваться всеми потоковыми источниками и приёмниками в приложении.

```python
from typed_flow.protocols import BaseThreadSource, BaseThreadProcessor, BaseThreadSink


class MySource(BaseThreadSource[MyEvent]):
    def __call__(self) -> MyEvent | None: ...


class MyProcessor(BaseThreadProcessor[MyEvent, MyResult]):
    def __call__(self, event: MyEvent) -> MyResult | None: ...


class MySink(BaseThreadSink[MyResult]):
    def __call__(self, event: MyResult) -> None: ...
```

## Процессные варианты (Process)

Используют `multiprocessing`. Каждый процесс имеет собственную память.

**Когда использовать:**
- CPU-bound операции (тяжёлые вычисления, ML-инференс на CPU)
- Обход GIL для настоящего параллелизма
- Изоляция от падений (один процесс не роняет другие)

**Redis-клиент:** нельзя передавать между процессами. Процессные источники и приёмники создают собственный клиент при первом вызове `__call__` внутри дочернего процесса.

```python
from typed_flow.connectors import RedisProcessSource, RedisProcessSink

# API идентичен потоковому варианту
source = RedisProcessSource[MyEvent](redis=redis_cfg, source=source_cfg)
sink = RedisProcessSink[MyEvent, MyResult](redis=redis_cfg, sink=sink_cfg)
```

## Пулы (Pool)

### ThreadPoolProcessor

N потоков обрабатывают общую входную очередь. Подходит для I/O-bound обработки с высоким параллелизмом.

```python
from typed_flow.protocols import BaseThreadPoolProcessor


class ParallelProcessor(BaseThreadPoolProcessor[InputEvent, OutputEvent]):
    def __call__(self, event: InputEvent) -> OutputEvent | None:
        # Вызывается из любого потока пула
        return process(event)
```

### ProcessPoolProcessor

N процессов обрабатывают общую очередь. Настоящий CPU-параллелизм.

```python
from typed_flow.protocols import BaseProcessPoolProcessor


class HeavyProcessor(BaseProcessPoolProcessor[InputEvent, OutputEvent]):
    def __call__(self, event: InputEvent) -> OutputEvent | None:
        # Вызывается в отдельном OS-процессе
        return heavy_computation(event)
```

## Жизненный цикл

Каждый узел имеет `stop: threading.Event`, который фреймворк устанавливает при завершении:

```python
class MySource(BaseThreadSource[MyEvent]):
    def __call__(self) -> MyEvent | None:
        if self.stop.is_set():
            return None
        # ... обычная логика
```

Фреймворк:
1. Создаёт `stop` event для каждого узла
2. Запускает узлы в выделенных потоках/процессах
3. Мониторит здоровье — перезапускает упавшие узлы
4. При `shutdown()` устанавливает `stop` для всех узлов

## Выбор варианта: решающее дерево

```
Нужен настоящий CPU-параллелизм?
├── Да → ProcessPoolProcessor или BaseProcessSource/Sink
│         (помните: каждый процесс создаёт свой Redis-клиент)
└── Нет → Потоковые варианты
          ├── Один экземпляр → BaseThreadSource/Processor/Sink
          └── Много экземпляров → BaseThreadPoolProcessor
```
