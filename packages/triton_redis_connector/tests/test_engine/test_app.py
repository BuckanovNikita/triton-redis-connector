"""Tests for Application lifecycle — startup, shutdown, inline dispatch, metrics."""

from __future__ import annotations

import threading
import time
from typing import Annotated, ClassVar

import pytest
from triton_redis_connector.config.schema import AppConfig, QueueConfig
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.metric_aggregator import Counter, Gauge, MetricAggregator
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry
from triton_redis_connector.metrics.buffer import MetricBuffer


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


class EventWithCountedMetric:
    _metrics_: ClassVar[list[Counter]] = [Counter("counted_metric")]


class SimpleSource:
    def __init__(self) -> None:
        self._events: list[EventA] = []

    def add_event(self, event: EventA) -> None:
        self._events.append(event)

    def __call__(self) -> EventA | None:
        if self._events:
            return self._events.pop(0)
        return None


class SimpleProcessor:
    def __init__(self) -> None:
        self.received: list[EventA] = []

    def __call__(self, event: EventA) -> EventB | None:
        self.received.append(event)
        return EventB()


class SimpleSink:
    def __init__(self) -> None:
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


class TestInlineDispatch:
    def test_dispatches_events_through_pipeline(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()

        event_a = EventA()
        source.add_event(event_a)

        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        app = Application(config=config, registry=registry, queue_bus=queue_bus)

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.1)
        app.shutdown()
        run_thread.join(timeout=2.0)

        assert len(proc.received) == 1
        assert proc.received[0] is event_a
        assert len(sink.received) == 1


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


class TestStatsMetrics:
    def test_log_stats_emits_zero_value_metrics_when_idle(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)
        app._log_stats()

        events = buffer.flush()
        measures = {e.measure for e in events}

        assert "stats_processed" in measures
        assert "stats_in_progress" in measures
        assert "stats_errors" in measures
        assert "node_events_total" in measures
        assert "node_events_interval" in measures
        assert "node_errors" in measures
        assert "sink_write_total" in measures

        for event in events:
            if event.measure in ("stats_processed", "stats_in_progress", "stats_errors"):
                assert event.value == 0.0

    def test_log_stats_emits_per_node_for_all_nodes(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)
        app._log_stats()

        events = buffer.flush()
        node_total_events = [e for e in events if e.measure == "node_events_total"]
        assert len(node_total_events) == 3

    def test_log_stats_interval_shows_delta(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)

        app._stats_per_node["proc"] = 5
        app._log_stats()
        buffer.flush()

        app._stats_per_node["proc"] = 8
        app._log_stats()
        events = buffer.flush()

        interval_events = [e for e in events if e.measure == "node_events_interval"]
        assert any(e.value == 3.0 for e in interval_events)

    def test_node_processing_ms_emitted_on_dispatch(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        source.add_event(EventA())
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.1)
        app.shutdown()
        run_thread.join(timeout=2.0)

        events = buffer.flush()
        processing_events = [e for e in events if e.measure == "node_processing_ms"]
        assert len(processing_events) >= 1
        assert all(e.value >= 0.0 for e in processing_events)

    def test_queue_depth_metrics_emitted(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        source = SimpleSource()
        proc = SimpleProcessor()
        sink = SimpleSink()
        registry.register("src", source)
        registry.register("proc", proc)
        registry.register("sink", sink)

        queue_bus.get_queue("test:input")
        queue_bus.put("test:input", "item")

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)
        app._log_stats()

        events = buffer.flush()
        depth_events = [e for e in events if e.measure == "queue_depth"]
        assert len(depth_events) >= 1
        assert any(e.value == 1.0 for e in depth_events)


class TestWrapperAggregator:
    def test_wrapper_records_metrics_on_call(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        aggregator = MetricAggregator()
        wrapper: DetachedThreadProcessor[str, str] = DetachedThreadProcessor(lambda x: x.upper(), aggregator=aggregator)
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

        wrapper: DetachedThreadProcessor[int, int] = DetachedThreadProcessor(lambda x: x * 2)
        assert wrapper("5") == "55"  # type: ignore[arg-type]


class TestAnnotatedMetricRules:
    def test_counter_incremented_on_matching_result(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        class CountedProcessor:
            def __call__(self, event: EventA) -> EventWithCounter | None:
                return EventWithCounter()

        class CounterSink:
            def __call__(self, event: EventWithCounter) -> None:
                pass

        source = SimpleSource()
        source.add_event(EventA())
        registry.register("src", source)
        registry.register("proc", CountedProcessor())
        registry.register("sink", CounterSink())

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.1)
        app.shutdown()
        run_thread.join(timeout=2.0)

        assert app._stats_counters.get("my_counter") == 1

    def test_gauge_emitted_on_matching_result(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        class GaugeProcessor:
            def __call__(self, event: EventA) -> EventWithValue | None:
                return EventWithValue(latency=42.5)

        class ValueSink:
            def __call__(self, event: EventWithValue) -> None:
                pass

        source = SimpleSource()
        source.add_event(EventA())
        registry.register("src", source)
        registry.register("proc", GaugeProcessor())
        registry.register("sink", ValueSink())

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.1)
        app.shutdown()
        run_thread.join(timeout=2.0)

        events = buffer.flush()
        gauge_events = [e for e in events if e.measure == "my_gauge"]
        assert len(gauge_events) >= 1
        assert gauge_events[0].value == 42.5

    def test_zero_value_counters_emitted_for_annotated_processors(
        self, config: AppConfig, registry: PipelineRegistry, queue_bus: QueueBus
    ) -> None:
        class CountedProcessor:
            def __call__(self, event: EventA) -> EventWithCountedMetric | None:
                return EventWithCountedMetric()

        class CountedSink:
            def __call__(self, event: EventWithCountedMetric) -> None:
                pass

        source = SimpleSource()
        registry.register("src", source)
        registry.register("proc", CountedProcessor())
        registry.register("sink", CountedSink())

        buffer = MetricBuffer()
        app = Application(config=config, registry=registry, queue_bus=queue_bus, metric_buffer=buffer)
        app._log_stats()

        events = buffer.flush()
        counter_events = [e for e in events if e.measure == "counted_metric"]
        assert len(counter_events) == 1
        assert counter_events[0].value == 0.0
