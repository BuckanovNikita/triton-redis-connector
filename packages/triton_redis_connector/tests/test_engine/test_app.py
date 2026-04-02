"""Tests for Application lifecycle — startup, shutdown, parallel dispatch."""

from __future__ import annotations

import threading
import time
from typing import Annotated, ClassVar

import pytest
from triton_redis_connector.config.schema import AppConfig, QueueConfig
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.metric_aggregator import MetricAggregator
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry
from triton_redis_connector.metrics.buffer import MetricBuffer
from triton_redis_connector.protocols import (
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
    Counter,
    Gauge,
    GroupBy,
)


class EventA:
    pass


class EventB:
    pass


class EventWithCounter:
    _metrics_: ClassVar[list[Counter]] = [Counter("my_counter")]


class EventWithValue:
    latency: Annotated[float, Gauge("my_gauge")]

    def __init__(self, latency: float) -> None:
        self.latency = latency


class EventWithGroupBy:
    camera_no: Annotated[str, GroupBy("camera_events", window_s=60)]

    def __init__(self, camera_no: str) -> None:
        self.camera_no = camera_no


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


class TestWrapperAggregator:
    def test_wrapper_records_metrics_on_call(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        def _upper(x: str) -> str:
            return x.upper()

        aggregator = MetricAggregator()
        wrapper: DetachedThreadProcessor[str, str] = DetachedThreadProcessor(_upper, aggregator=aggregator)
        result = wrapper("hello")
        assert result == "HELLO"
        snap = aggregator.snapshot()
        assert snap.call_count == 1
        assert snap.total_processing_ms > 0

    def test_wrapper_records_error_on_exception(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        def failing(_: str) -> str:
            msg = "boom"
            raise ValueError(msg)

        aggregator = MetricAggregator()
        wrapper: DetachedThreadProcessor[str, str] = DetachedThreadProcessor(failing, aggregator=aggregator)
        with pytest.raises(ValueError, match="boom"):
            wrapper("x")
        snap = aggregator.snapshot()
        assert snap.error_count == 1

    def test_wrapper_records_empty_for_none_result(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadSource

        aggregator = MetricAggregator()
        wrapper: DetachedThreadSource[str] = DetachedThreadSource(lambda: None, aggregator=aggregator)
        result = wrapper()
        assert result is None
        snap = aggregator.snapshot()
        assert snap.empty_count == 1

    def test_wrapper_without_aggregator_passes_through(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        def _double(x: int) -> int:
            return x * 2

        wrapper: DetachedThreadProcessor[int, int] = DetachedThreadProcessor(_double)
        assert wrapper("5") == "55"  # type: ignore[arg-type]


class TestWrapperMetricExtraction:
    def test_counter_extracted_to_buffer(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        buffer = MetricBuffer()

        def produce_counted(_: EventA) -> EventWithCounter:
            return EventWithCounter()

        wrapper: DetachedThreadProcessor[EventA, EventWithCounter] = DetachedThreadProcessor(
            produce_counted, metric_recorder=buffer, component_name="test"
        )
        wrapper(EventA())

        events = buffer.flush()
        counter_events = [e for e in events if e.measure == "my_counter"]
        assert len(counter_events) == 1
        assert counter_events[0].value == 1.0

    def test_gauge_extracted_to_buffer(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        buffer = MetricBuffer()

        def produce_gauged(_: EventA) -> EventWithValue:
            return EventWithValue(latency=42.5)

        wrapper: DetachedThreadProcessor[EventA, EventWithValue] = DetachedThreadProcessor(
            produce_gauged, metric_recorder=buffer, component_name="test"
        )
        wrapper(EventA())

        events = buffer.flush()
        gauge_events = [e for e in events if e.measure == "my_gauge"]
        assert len(gauge_events) == 1
        assert gauge_events[0].value == 42.5

    def test_group_by_extracted_to_buffer(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        buffer = MetricBuffer()

        def produce_grouped(_: EventA) -> EventWithGroupBy:
            return EventWithGroupBy(camera_no="cam_2")

        wrapper: DetachedThreadProcessor[EventA, EventWithGroupBy] = DetachedThreadProcessor(
            produce_grouped, metric_recorder=buffer, component_name="test"
        )
        wrapper(EventA())

        events = buffer.flush()
        group_events = [e for e in events if e.measure == "camera_events:cam_2"]
        assert len(group_events) == 1
        assert group_events[0].value == 1.0

    def test_node_processing_ms_emitted(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        buffer = MetricBuffer()

        def _upper(x: str) -> str:
            return x.upper()

        wrapper: DetachedThreadProcessor[str, str] = DetachedThreadProcessor(
            _upper, metric_recorder=buffer, component_name="test"
        )
        wrapper("hello")

        events = buffer.flush()
        processing_events = [e for e in events if e.measure == "node_processing_ms"]
        assert len(processing_events) == 1
        assert processing_events[0].value >= 0.0

    def test_no_metrics_emitted_when_result_is_none(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        buffer = MetricBuffer()

        def _returns_none(_: str) -> None:
            return None

        wrapper: DetachedThreadProcessor[str, str | None] = DetachedThreadProcessor(
            _returns_none, metric_recorder=buffer, component_name="test"
        )
        wrapper("hello")

        events = buffer.flush()
        assert len(events) == 0

    def test_source_wrapper_extracts_metrics(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadSource

        buffer = MetricBuffer()

        wrapper: DetachedThreadSource[EventWithCounter] = DetachedThreadSource(
            EventWithCounter, metric_recorder=buffer, component_name="test"
        )
        wrapper()

        events = buffer.flush()
        counter_events = [e for e in events if e.measure == "my_counter"]
        assert len(counter_events) == 1

    def test_sink_wrapper_extracts_metrics(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadSink

        buffer = MetricBuffer()
        received: list[EventWithValue] = []

        def sink_fn(event: EventWithValue) -> EventWithValue:
            received.append(event)
            return event

        wrapper: DetachedThreadSink[EventWithValue] = DetachedThreadSink(
            sink_fn, metric_recorder=buffer, component_name="test"
        )
        wrapper(EventWithValue(latency=99.9))

        events = buffer.flush()
        gauge_events = [e for e in events if e.measure == "my_gauge"]
        assert len(gauge_events) == 1
        assert gauge_events[0].value == 99.9
