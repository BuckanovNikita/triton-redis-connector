"""Tests for Protocol runtime_checkable isinstance checks on all 9 variants."""

from __future__ import annotations

import threading
from typing import Any

from triton_redis_connector.protocols import (
    AggregationRuleProtocol,
    Envelope,
    ObserverProtocol,
    Processor,
    ProcessorFactory,
    ProcessPoolProcessor,
    ProcessSource,
    Sink,
    Source,
    ThreadPoolProcessor,
    ThreadProcessor,
    ThreadSink,
    ThreadSource,
)


class DummySource:
    def __call__(self) -> str | None:
        return "event"


class DummyThreadSource:
    stop = threading.Event()

    def __call__(self) -> str | None:
        return "event"


class DummyProcessSource:
    stop = threading.Event()

    def __call__(self) -> int | None:
        return 42


class DummyProcessor:
    def __call__(self, event: str) -> int | None:
        return len(event)


class DummyThreadProcessor:
    stop = threading.Event()

    def __call__(self, event: str) -> int | None:
        return len(event)


class DummyThreadPoolProcessor:
    stop = threading.Event()

    def __call__(self, event: str) -> int | None:
        return len(event)


class DummyProcessPoolProcessor:
    stop = threading.Event()

    def __call__(self, event: bytes) -> str | None:
        return event.decode()


class DummySink:
    def __call__(self, _event: str) -> None:
        return None


class DummyThreadSink:
    stop = threading.Event()

    def __call__(self, _event: str) -> Any | None:
        return None


class DummyProcessorFactory:
    def create_shared(self) -> dict[str, int]:
        return {"counter": 0}

    def create_processor(self, _shared: dict[str, int]) -> DummyProcessor:
        return DummyProcessor()


class DummyObserver:
    def observe(self, _envelope: Envelope[str]) -> None:
        pass


class DummyAggregationRule:
    @property
    def metric_name(self) -> str:
        return "request_count"

    @property
    def window_seconds(self) -> int:
        return 60

    def extract_key(self, event: str) -> str:
        return event

    def aggregate(self, count: int) -> float:
        return float(count)


class TestSourceProtocols:
    def test_source_isinstance(self) -> None:
        assert isinstance(DummySource(), Source)

    def test_thread_source_isinstance(self) -> None:
        assert isinstance(DummyThreadSource(), ThreadSource)

    def test_process_source_isinstance(self) -> None:
        assert isinstance(DummyProcessSource(), ProcessSource)

    def test_source_without_call_not_instance(self) -> None:
        class NoCall:
            pass

        assert not isinstance(NoCall(), Source)

    def test_thread_source_without_stop_not_instance(self) -> None:
        class NoStop:
            def __call__(self) -> str | None:
                return None

        assert not isinstance(NoStop(), ThreadSource)


class TestProcessorProtocols:
    def test_processor_isinstance(self) -> None:
        assert isinstance(DummyProcessor(), Processor)

    def test_thread_processor_isinstance(self) -> None:
        assert isinstance(DummyThreadProcessor(), ThreadProcessor)

    def test_thread_pool_processor_isinstance(self) -> None:
        assert isinstance(DummyThreadPoolProcessor(), ThreadPoolProcessor)

    def test_process_pool_processor_isinstance(self) -> None:
        assert isinstance(DummyProcessPoolProcessor(), ProcessPoolProcessor)

    def test_processor_without_call_not_instance(self) -> None:
        class NoCall:
            pass

        assert not isinstance(NoCall(), Processor)

    def test_thread_processor_without_stop_not_instance(self) -> None:
        class NoStop:
            def __call__(self, _event: str) -> int | None:
                return None

        assert not isinstance(NoStop(), ThreadProcessor)


class TestSinkProtocols:
    def test_sink_isinstance(self) -> None:
        assert isinstance(DummySink(), Sink)

    def test_thread_sink_isinstance(self) -> None:
        assert isinstance(DummyThreadSink(), ThreadSink)

    def test_sink_without_call_not_instance(self) -> None:
        class NoCall:
            pass

        assert not isinstance(NoCall(), Sink)

    def test_thread_sink_without_stop_not_instance(self) -> None:
        class NoStop:
            def __call__(self, _event: str) -> None:
                return None

        assert not isinstance(NoStop(), ThreadSink)


class TestProcessorFactoryProtocol:
    def test_factory_isinstance(self) -> None:
        assert isinstance(DummyProcessorFactory(), ProcessorFactory)

    def test_factory_without_create_shared_not_instance(self) -> None:
        class NoCreateShared:
            def create_processor(self, _shared: object) -> DummyProcessor:
                return DummyProcessor()

        assert not isinstance(NoCreateShared(), ProcessorFactory)

    def test_factory_without_create_processor_not_instance(self) -> None:
        class NoCreateProcessor:
            def create_shared(self) -> dict[str, int]:
                return {}

        assert not isinstance(NoCreateProcessor(), ProcessorFactory)


class TestObserverProtocol:
    def test_observer_isinstance(self) -> None:
        assert isinstance(DummyObserver(), ObserverProtocol)

    def test_observer_without_observe_not_instance(self) -> None:
        class NoObserve:
            pass

        assert not isinstance(NoObserve(), ObserverProtocol)


class TestAggregationRuleProtocol:
    def test_aggregation_rule_isinstance(self) -> None:
        assert isinstance(DummyAggregationRule(), AggregationRuleProtocol)

    def test_aggregation_rule_functional(self) -> None:
        rule = DummyAggregationRule()
        assert rule.metric_name == "request_count"
        assert rule.window_seconds == 60
        assert rule.extract_key("test") == "test"
        assert rule.aggregate(5) == 5.0

    def test_aggregation_rule_without_methods_not_instance(self) -> None:
        class Incomplete:
            pass

        assert not isinstance(Incomplete(), AggregationRuleProtocol)


class TestCrossVariantStructuralTyping:
    def test_source_matches_processor_structurally(self) -> None:
        assert isinstance(DummySource(), Processor)

    def test_processor_matches_source_structurally(self) -> None:
        assert isinstance(DummyProcessor(), Source)

    def test_thread_source_is_also_process_source(self) -> None:
        ts = DummyThreadSource()
        assert isinstance(ts, ProcessSource)

    def test_sink_matches_source_structurally(self) -> None:
        assert isinstance(DummySink(), Source)

    def test_source_without_stop_not_thread_source(self) -> None:
        assert not isinstance(DummySource(), ThreadSource)

    def test_processor_without_stop_not_thread_processor(self) -> None:
        assert not isinstance(DummyProcessor(), ThreadProcessor)
