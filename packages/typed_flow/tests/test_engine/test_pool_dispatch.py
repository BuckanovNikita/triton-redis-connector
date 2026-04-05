"""Tests for pool variant dispatch — thread-pool and process-pool processors."""

import threading
import time
from typing import Any

import pytest
from typed_flow.config.schema import AppConfig, QueueConfig
from typed_flow.engine.app import Application
from typed_flow.engine.queue_bus import QueueBus
from typed_flow.engine.registry import PipelineRegistry
from typed_flow.errors.models import PipelineError
from typed_flow.metrics.buffer import MetricBuffer
from typed_flow.protocols import (
    BaseProcessPoolProcessor,
    BaseThreadPoolProcessor,
    BaseThreadSink,
    BaseThreadSource,
)


class EventA:
    pass


class EventB:
    pass


class PoolSource(BaseThreadSource[EventA]):
    def __init__(self) -> None:
        super().__init__()
        self._events: list[EventA] = []

    def add_events(self, n: int) -> None:
        self._events.extend(EventA() for _ in range(n))

    def __call__(self) -> EventA | None:
        if self._events:
            return self._events.pop(0)
        return None


class CollectingSink(BaseThreadSink[EventB]):
    def __init__(self) -> None:
        super().__init__()
        self.received: list[EventB] = []
        self._lock = threading.Lock()

    def __call__(self, event: EventB) -> None:
        with self._lock:
            self.received.append(event)


class ErrorSink(BaseThreadSink[PipelineError]):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, event: PipelineError) -> Any | None:
        return None


class ThreadPoolProc(BaseThreadPoolProcessor[EventA, EventB]):
    def __init__(self, num_workers: int = 4) -> None:
        super().__init__(num_workers=num_workers)
        self.call_count = 0
        self._lock = threading.Lock()

    def __call__(self, event: EventA) -> EventB | None:
        with self._lock:
            self.call_count += 1
        return EventB()


class ProcessPoolProc(BaseProcessPoolProcessor[EventA, EventB]):
    def __init__(self, num_workers: int = 2) -> None:
        super().__init__(num_workers=num_workers)

    def __call__(self, event: EventA) -> EventB | None:
        return EventB()


class UnpicklableProc(BaseProcessPoolProcessor[EventA, EventB]):
    def __init__(self) -> None:
        super().__init__(num_workers=2)
        self._lock = threading.Lock()

    def __call__(self, event: EventA) -> EventB | None:
        return EventB()


def _build_pipeline(
    source: PoolSource,
    proc: object,
    sink: CollectingSink,
) -> Application:
    config = AppConfig(queues=QueueConfig(maxsize=200))
    registry = PipelineRegistry()
    queue_bus = QueueBus(config=config.queues)
    buffer = MetricBuffer()

    registry.register("src", source)
    registry.register("proc", proc)
    registry.register("sink", sink)
    registry.register("error_sink", ErrorSink())

    return Application(
        config=config,
        registry=registry,
        queue_bus=queue_bus,
        metric_buffer=buffer,
    )


def _run_and_wait(app: Application, source: PoolSource, sink: CollectingSink, n_events: int, timeout: float) -> None:
    source.add_events(n_events)
    run_thread = threading.Thread(target=app.run, daemon=True)
    run_thread.start()
    deadline = time.monotonic() + timeout
    while len(sink.received) < n_events and time.monotonic() < deadline:
        time.sleep(0.01)
    app.shutdown()
    run_thread.join(timeout=2.0)


class TestThreadPoolDispatch:
    def test_thread_pool_processes_all_events(self) -> None:
        source = PoolSource()
        proc = ThreadPoolProc(num_workers=4)
        sink = CollectingSink()
        app = _build_pipeline(source, proc, sink)

        _run_and_wait(app, source, sink, n_events=20, timeout=3.0)
        assert len(sink.received) == 20

    def test_thread_pool_spawns_worker_threads(self) -> None:
        source = PoolSource()
        proc = ThreadPoolProc(num_workers=4)
        sink = CollectingSink()
        app = _build_pipeline(source, proc, sink)

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.1)

        worker_keys = [k for k in app._parallel_threads if k.startswith("proc:worker-")]  # pyright: ignore[reportPrivateUsage]
        app.shutdown()
        run_thread.join(timeout=2.0)

        assert len(worker_keys) == 4

    def test_thread_pool_no_duplicates(self) -> None:
        source = PoolSource()
        proc = ThreadPoolProc(num_workers=4)
        sink = CollectingSink()
        app = _build_pipeline(source, proc, sink)

        _run_and_wait(app, source, sink, n_events=50, timeout=5.0)
        assert len(sink.received) == 50
        assert proc.call_count == 50


class TestProcessPoolDispatch:
    def test_process_pool_processes_all_events(self) -> None:
        source = PoolSource()
        proc = ProcessPoolProc(num_workers=2)
        sink = CollectingSink()
        app = _build_pipeline(source, proc, sink)

        _run_and_wait(app, source, sink, n_events=10, timeout=5.0)
        assert len(sink.received) == 10

    def test_process_pool_clean_shutdown(self) -> None:
        source = PoolSource()
        proc = ProcessPoolProc(num_workers=2)
        sink = CollectingSink()
        app = _build_pipeline(source, proc, sink)

        source.add_events(5)
        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.3)
        app.shutdown()
        run_thread.join(timeout=5.0)

        assert not run_thread.is_alive()


class TestPicklabilityValidation:
    def test_unpicklable_node_fails_at_registration(self) -> None:
        registry = PipelineRegistry()
        with pytest.raises(TypeError, match="not picklable"):
            registry.register("bad", UnpicklableProc())


class TestPoolShutdown:
    def test_shutdown_under_load(self) -> None:
        source = PoolSource()
        proc = ThreadPoolProc(num_workers=4)
        sink = CollectingSink()
        app = _build_pipeline(source, proc, sink)

        source.add_events(100)
        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()
        time.sleep(0.05)
        app.shutdown()
        run_thread.join(timeout=3.0)

        assert not run_thread.is_alive()
