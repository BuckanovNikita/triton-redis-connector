"""Tests for PipelineRegistry — DAG construction, validation, dispatch order, and metric extraction."""

from __future__ import annotations

import threading
from typing import Annotated, ClassVar

import pytest
from triton_redis_connector.engine.metric_aggregator import Counter, Gauge
from triton_redis_connector.engine.registry import ExecutionVariant, PipelineRegistry


class EventA:
    pass


class EventB:
    pass


class EventC:
    pass


class SimpleSource:
    def __call__(self) -> EventA | None:
        return EventA()


class ThreadedSource:
    stop = threading.Event()

    def __call__(self) -> EventA | None:
        return EventA()


class UnionSource:
    def __call__(self) -> EventA | EventB | None:
        return EventA()


class SimpleProcessor:
    def __call__(self, event: EventA) -> EventB | None:
        return EventB()


class MultiOutputProcessor:
    def __call__(self, event: EventA) -> EventB | EventC | None:
        return EventB()


class SimpleSink:
    def __call__(self, event: EventB) -> None:
        return None


class ThreadedSink:
    stop = threading.Event()

    def __call__(self, event: EventB) -> None:
        return None


class ChainProcessor:
    def __call__(self, event: EventB) -> EventC | None:
        return EventC()


class EventCSink:
    def __call__(self, event: EventC) -> None:
        return None


@pytest.fixture
def registry() -> PipelineRegistry:
    return PipelineRegistry()


class TestVariantDetection:
    def test_detects_inline_source(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        node = registry.get_node("src")
        assert node is not None
        assert node.variant == ExecutionVariant.SOURCE

    def test_detects_thread_source(self, registry: PipelineRegistry) -> None:
        registry.register("src", ThreadedSource())
        node = registry.get_node("src")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_SOURCE

    def test_detects_inline_processor(self, registry: PipelineRegistry) -> None:
        registry.register("proc", SimpleProcessor())
        node = registry.get_node("proc")
        assert node is not None
        assert node.variant == ExecutionVariant.PROCESSOR

    def test_detects_inline_sink(self, registry: PipelineRegistry) -> None:
        registry.register("sink", SimpleSink())
        node = registry.get_node("sink")
        assert node is not None
        assert node.variant == ExecutionVariant.SINK

    def test_detects_thread_sink(self, registry: PipelineRegistry) -> None:
        registry.register("sink", ThreadedSink())
        node = registry.get_node("sink")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_SINK

    def test_raises_on_invalid_node(self, registry: PipelineRegistry) -> None:
        with pytest.raises(TypeError, match="Cannot detect execution variant"):
            registry.register("bad", object())


class TestTypeExtraction:
    def test_source_output_types(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        node = registry.get_node("src")
        assert node is not None
        assert EventA in node.output_types

    def test_union_source_output_types(self, registry: PipelineRegistry) -> None:
        registry.register("src", UnionSource())
        node = registry.get_node("src")
        assert node is not None
        assert EventA in node.output_types
        assert EventB in node.output_types

    def test_processor_input_output_types(self, registry: PipelineRegistry) -> None:
        registry.register("proc", SimpleProcessor())
        node = registry.get_node("proc")
        assert node is not None
        assert EventA in node.input_types
        assert EventB in node.output_types

    def test_multi_output_processor(self, registry: PipelineRegistry) -> None:
        registry.register("proc", MultiOutputProcessor())
        node = registry.get_node("proc")
        assert node is not None
        assert EventA in node.input_types
        assert EventB in node.output_types
        assert EventC in node.output_types

    def test_sink_input_types(self, registry: PipelineRegistry) -> None:
        registry.register("sink", SimpleSink())
        node = registry.get_node("sink")
        assert node is not None
        assert EventB in node.input_types


class TestValidation:
    def test_valid_pipeline_passes(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        registry.register("proc", SimpleProcessor())
        registry.register("sink", SimpleSink())
        errors = registry.validate()
        assert errors == []

    def test_missing_consumer_reports_error(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        errors = registry.validate()
        assert any("no node consumes it" in e for e in errors)

    def test_missing_producer_reports_error(self, registry: PipelineRegistry) -> None:
        registry.register("sink", SimpleSink())
        errors = registry.validate()
        assert any("no node produces it" in e for e in errors)

    def test_disconnected_subgraph_reports_error(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        registry.register("proc", SimpleProcessor())
        registry.register("sink", SimpleSink())
        registry.register("sink2", EventCSink())
        errors = registry.validate()
        error_text = " ".join(errors)
        assert "disconnected" in error_text or "no node produces" in error_text

    def test_no_sources_reports_error(self, registry: PipelineRegistry) -> None:
        registry.register("proc", SimpleProcessor())
        registry.register("sink", SimpleSink())
        errors = registry.validate()
        assert any("No source" in e or "no node produces" in e for e in errors)

    def test_union_output_all_consumed_valid(self, registry: PipelineRegistry) -> None:
        registry.register("src", UnionSource())

        class SinkA:
            def __call__(self, event: EventA) -> None:
                return None

        class SinkB:
            def __call__(self, event: EventB) -> None:
                return None

        registry.register("sink_a", SinkA())
        registry.register("sink_b", SinkB())
        errors = registry.validate()
        assert errors == []

    def test_full_chain_valid(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        registry.register("proc1", SimpleProcessor())
        registry.register("proc2", ChainProcessor())
        registry.register("sink", EventCSink())
        errors = registry.validate()
        assert errors == []


class TestDispatchOrder:
    def test_returns_inline_nodes_only(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        registry.register("proc", SimpleProcessor())
        registry.register("sink", SimpleSink())
        registry.register("thread_src", ThreadedSource())
        order = registry.get_dispatch_order()
        names = [n.name for n in order]
        assert "thread_src" not in names
        assert "src" in names

    def test_topological_order_source_before_processor(self, registry: PipelineRegistry) -> None:
        registry.register("proc", SimpleProcessor())
        registry.register("src", SimpleSource())
        registry.register("sink", SimpleSink())
        order = registry.get_dispatch_order()
        names = [n.name for n in order]
        assert names.index("src") < names.index("proc")
        assert names.index("proc") < names.index("sink")

    def test_empty_registry_returns_empty(self, registry: PipelineRegistry) -> None:
        assert registry.get_dispatch_order() == []

    def test_chain_ordering(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        registry.register("proc1", SimpleProcessor())
        registry.register("proc2", ChainProcessor())
        registry.register("sink", EventCSink())
        order = registry.get_dispatch_order()
        names = [n.name for n in order]
        assert names.index("src") < names.index("proc1")
        assert names.index("proc1") < names.index("proc2")
        assert names.index("proc2") < names.index("sink")


class TestNodeAccess:
    def test_get_node_returns_registered(self, registry: PipelineRegistry) -> None:
        src = SimpleSource()
        registry.register("src", src)
        node = registry.get_node("src")
        assert node is not None
        assert node.instance is src

    def test_get_node_returns_none_for_missing(self, registry: PipelineRegistry) -> None:
        assert registry.get_node("nonexistent") is None

    def test_nodes_property(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        registry.register("proc", SimpleProcessor())
        assert len(registry.nodes) == 2
        assert "src" in registry.nodes
        assert "proc" in registry.nodes


class EventWithCounter:
    _metrics_: ClassVar[list[Counter]] = [Counter("success")]


class EventWithFailCounter:
    _metrics_: ClassVar[list[Counter]] = [Counter("fail")]


class EventWithAlpha:
    _metrics_: ClassVar[list[Counter]] = [Counter("alpha")]


class EventWithBeta:
    _metrics_: ClassVar[list[Counter]] = [Counter("beta")]


class EventWithGaugeField:
    latency: Annotated[float, Gauge("latency")]

    def __init__(self, latency: float) -> None:
        self.latency = latency


class TestAnnotatedMetricExtraction:
    def test_counter_from_event_class(self, registry: PipelineRegistry) -> None:
        class Proc:
            def __call__(self, event: EventA) -> EventWithCounter | None:
                return EventWithCounter()

        class CounterSink:
            def __call__(self, event: EventWithCounter) -> None:
                pass

        registry.register("src", SimpleSource())
        registry.register("proc", Proc())
        registry.register("sink", CounterSink())
        assert "success" in registry.all_counter_names

    def test_multiple_counters_from_event_types(self, registry: PipelineRegistry) -> None:
        class ProcOk:
            def __call__(self, event: EventA) -> EventWithCounter | EventWithFailCounter | None:
                return EventWithCounter()

        class CounterSink:
            def __call__(self, event: EventWithCounter) -> None:
                pass

        class FailSink:
            def __call__(self, event: EventWithFailCounter) -> None:
                pass

        registry.register("src", SimpleSource())
        registry.register("proc", ProcOk())
        registry.register("sink_ok", CounterSink())
        registry.register("sink_fail", FailSink())
        assert "success" in registry.all_counter_names
        assert "fail" in registry.all_counter_names

    def test_annotated_does_not_break_output_types(self, registry: PipelineRegistry) -> None:
        class AnnotatedProc:
            def __call__(self, event: EventA) -> Annotated[EventB, Counter("x")] | None:
                return EventB()

        registry.register("proc", AnnotatedProc())
        node = registry.get_node("proc")
        assert node is not None
        assert EventB in node.output_types

    def test_all_counter_names(self, registry: PipelineRegistry) -> None:
        class Proc1:
            def __call__(self, event: EventA) -> EventWithAlpha | None:
                return EventWithAlpha()

        class Proc2:
            def __call__(self, event: EventWithAlpha) -> EventWithBeta | None:
                return EventWithBeta()

        class AlphaSink:
            def __call__(self, event: EventWithAlpha) -> None:
                pass

        class BetaSink:
            def __call__(self, event: EventWithBeta) -> None:
                pass

        registry.register("src", SimpleSource())
        registry.register("p1", Proc1())
        registry.register("p2", Proc2())
        registry.register("sink_a", AlphaSink())
        registry.register("sink_b", BetaSink())
        assert registry.all_counter_names == {"alpha", "beta"}

    def test_all_counter_names_empty_when_no_metrics(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        registry.register("proc", SimpleProcessor())
        registry.register("sink", SimpleSink())
        assert registry.all_counter_names == set()

    def test_gauge_extracted_from_event_field(self, registry: PipelineRegistry) -> None:
        from triton_redis_connector.engine.metric_aggregator import extract_field_gauges

        gauges = extract_field_gauges(EventWithGaugeField)
        assert len(gauges) == 1
        assert gauges[0] == ("latency", Gauge("latency"))
