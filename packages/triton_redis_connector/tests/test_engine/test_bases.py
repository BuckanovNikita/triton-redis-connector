"""Tests for optional base classes satisfying execution variant protocols."""

from __future__ import annotations

import threading

from triton_redis_connector.engine.bases import (
    BaseProcessPoolProcessor,
    BaseProcessSource,
    BaseThreadPoolProcessor,
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
)
from triton_redis_connector.protocols import (
    ProcessPoolProcessor,
    ProcessSource,
    ThreadPoolProcessor,
    ThreadProcessor,
    ThreadSink,
    ThreadSource,
)


class ConcreteThreadSource(BaseThreadSource[str]):
    def __call__(self) -> str | None:
        return "event"


class ConcreteProcessSource(BaseProcessSource[int]):
    def __call__(self) -> int | None:
        return 42


class ConcreteThreadProcessor(BaseThreadProcessor[str, int]):
    def __call__(self, event: str) -> int | None:
        return len(event)


class ConcreteThreadPoolProcessor(BaseThreadPoolProcessor[str, int]):
    def __call__(self, event: str) -> int | None:
        return len(event)


class ConcreteProcessPoolProcessor(BaseProcessPoolProcessor[str, int]):
    def __call__(self, event: str) -> int | None:
        return len(event)


class ConcreteThreadSink(BaseThreadSink[str]):
    def __call__(self, event: str) -> None:
        pass


class TestBaseThreadSource:
    def test_satisfies_protocol(self) -> None:
        source = ConcreteThreadSource()
        assert isinstance(source, ThreadSource)

    def test_stop_event_initialized(self) -> None:
        source = ConcreteThreadSource()
        assert isinstance(source.stop, threading.Event)
        assert not source.stop.is_set()

    def test_call_returns_event(self) -> None:
        source = ConcreteThreadSource()
        assert source() == "event"


class TestBaseProcessSource:
    def test_satisfies_protocol(self) -> None:
        source = ConcreteProcessSource()
        assert isinstance(source, ProcessSource)

    def test_stop_event_initialized(self) -> None:
        source = ConcreteProcessSource()
        assert isinstance(source.stop, threading.Event)

    def test_call_returns_event(self) -> None:
        source = ConcreteProcessSource()
        assert source() == 42


class TestBaseThreadProcessor:
    def test_satisfies_protocol(self) -> None:
        proc = ConcreteThreadProcessor()
        assert isinstance(proc, ThreadProcessor)

    def test_stop_event_initialized(self) -> None:
        proc = ConcreteThreadProcessor()
        assert isinstance(proc.stop, threading.Event)

    def test_call_transforms_event(self) -> None:
        proc = ConcreteThreadProcessor()
        assert proc("hello") == 5


class TestBaseThreadPoolProcessor:
    def test_satisfies_protocol(self) -> None:
        proc = ConcreteThreadPoolProcessor()
        assert isinstance(proc, ThreadPoolProcessor)

    def test_stop_event_initialized(self) -> None:
        proc = ConcreteThreadPoolProcessor()
        assert isinstance(proc.stop, threading.Event)

    def test_call_transforms_event(self) -> None:
        proc = ConcreteThreadPoolProcessor()
        assert proc("hello") == 5


class TestBaseProcessPoolProcessor:
    def test_satisfies_protocol(self) -> None:
        proc = ConcreteProcessPoolProcessor()
        assert isinstance(proc, ProcessPoolProcessor)

    def test_stop_event_initialized(self) -> None:
        proc = ConcreteProcessPoolProcessor()
        assert isinstance(proc.stop, threading.Event)

    def test_call_transforms_event(self) -> None:
        proc = ConcreteProcessPoolProcessor()
        assert proc("hello") == 5


class TestBaseThreadSink:
    def test_satisfies_protocol(self) -> None:
        sink = ConcreteThreadSink()
        assert isinstance(sink, ThreadSink)

    def test_stop_event_initialized(self) -> None:
        sink = ConcreteThreadSink()
        assert isinstance(sink.stop, threading.Event)

    def test_call_consumes_event(self) -> None:
        sink = ConcreteThreadSink()
        assert sink("hello") is None


class TestBaseWithBusinessInit:
    """Verify base classes work when subclass adds business-specific init logic."""

    def test_processor_with_config(self) -> None:
        class ConfiguredProcessor(BaseThreadProcessor[str, str]):
            def __init__(self, prefix: str) -> None:
                super().__init__()
                self._prefix = prefix

            def __call__(self, event: str) -> str | None:
                return f"{self._prefix}:{event}"

        proc = ConfiguredProcessor(prefix="test")
        assert isinstance(proc, ThreadProcessor)
        assert proc("data") == "test:data"
        assert isinstance(proc.stop, threading.Event)
