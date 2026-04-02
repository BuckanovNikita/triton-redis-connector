"""Tests for Application lifecycle — startup, shutdown, parallel dispatch."""

from __future__ import annotations

import threading
import time

import pytest
from triton_redis_connector.config.schema import AppConfig, QueueConfig
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry
from triton_redis_connector.metrics.buffer import MetricBuffer
from triton_redis_connector.protocols import (
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
)


class EventA:
    pass


class EventB:
    pass


class SimpleSource(BaseThreadSource[EventA]):
    def __init__(self) -> None:
        super().__init__()
        self._events: list[EventA] = []

    def add_event(self, event: EventA) -> None:
        self._events.append(event)

    def __call__(self) -> EventA | None:
        if self._events:
            return self._events.pop(0)
        return None


class SimpleProcessor(BaseThreadProcessor[EventA, EventB]):
    def __init__(self) -> None:
        super().__init__()
        self.received: list[EventA] = []

    def __call__(self, event: EventA) -> EventB | None:
        self.received.append(event)
        return EventB()


class SimpleSink(BaseThreadSink[EventB]):
    def __init__(self) -> None:
        super().__init__()
        self.received: list[EventB] = []

    def __call__(self, event: EventB) -> None:
        self.received.append(event)


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def queue_bus() -> QueueBus:
    return QueueBus(config=QueueConfig(maxsize=100))


@pytest.fixture
def registry() -> PipelineRegistry:
    return PipelineRegistry()


class TestApplicationStartup:
    def test_validates_graph_on_run(self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus) -> None:
        registry.register("proc", SimpleProcessor())
        app = Application(config=config, registry=registry, queue_bus=queue_bus)
        with pytest.raises(ValueError, match="validation error"):
            app.run()

    def test_valid_graph_starts_without_error(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        app = Application(config=config, registry=registry, queue_bus=queue_bus)

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.05)
        app.shutdown()
        run_thread.join(timeout=2.0)
        assert not run_thread.is_alive()


class TestShutdown:
    def test_shutdown_sets_event(self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        app = Application(config=config, registry=registry, queue_bus=queue_bus)

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.05)
        app.shutdown()
        run_thread.join(timeout=2.0)
        assert not run_thread.is_alive()

    def test_multiple_shutdown_calls_safe(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        app = Application(config=config, registry=registry, queue_bus=queue_bus)

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.05)
        app.shutdown()
        app.shutdown()
        run_thread.join(timeout=2.0)
        assert not run_thread.is_alive()


class TestLogStats:
    def test_log_stats_only_logs_text(self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)
        app._log_stats()  # pyright: ignore[reportPrivateUsage]

        events = buffer.flush()
        assert len(events) == 0
