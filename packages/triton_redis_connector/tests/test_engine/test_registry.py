"""Tests for PipelineRegistry — DAG construction, validation, and metric extraction."""

from __future__ import annotations

from typing import Annotated, ClassVar

import pytest
from typed_flow.engine.registry import ExecutionVariant, PipelineRegistry
from typed_flow.protocols import (
    BaseProcessPoolProcessor,
    BaseProcessSource,
    BaseThreadPoolProcessor,
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
    Counter,
    Gauge,
    ProcessPoolProcessor,
    ProcessSource,
    ThreadPoolProcessor,
    ThreadProcessor,
    ThreadSink,
    ThreadSource,
    WrappedNode,
)


class EventA:
    pass


class EventB:
    pass


class EventC:
    pass


class SimpleSource(BaseThreadSource[EventA]):
    def __call__(self) -> EventA | None:
        return EventA()


class UnionSource(BaseThreadSource[EventA | EventB]):
    def __call__(self) -> EventA | EventB | None:
        return EventA()


class SimpleProcessor(BaseThreadProcessor[EventA, EventB]):
    def __call__(self, event: EventA) -> EventB | None:
        return EventB()


class MultiOutputProcessor(BaseThreadProcessor[EventA, EventB | EventC]):
    def __call__(self, event: EventA) -> EventB | EventC | None:
        return EventB()


class SimpleSink(BaseThreadSink[EventB]):
    def __call__(self, event: EventB) -> None:
        return None


class ChainProcessor(BaseThreadProcessor[EventB, EventC]):
    def __call__(self, event: EventB) -> EventC | None:
        return EventC()


class EventCSink(BaseThreadSink[EventC]):
    def __call__(self, event: EventC) -> None:
        return None


@pytest.fixture
def registry() -> PipelineRegistry:
    return PipelineRegistry()


class TestVariantDetection:
    def test_detects_thread_source_via_base_class(self, registry: PipelineRegistry) -> None:
        registry.register("src", SimpleSource())
        node = registry.get_node("src")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_SOURCE

    def test_detects_thread_processor_via_base_class(self, registry: PipelineRegistry) -> None:
        registry.register("proc", SimpleProcessor())
        node = registry.get_node("proc")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_PROCESSOR

    def test_detects_thread_sink_via_base_class(self, registry: PipelineRegistry) -> None:
        registry.register("sink", SimpleSink())
        node = registry.get_node("sink")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_SINK

    def test_raises_on_non_callable(self, registry: PipelineRegistry) -> None:
        with pytest.raises(TypeError, match="not callable"):
            registry.register("bad", object())

    def test_raises_on_plain_callable_without_stop(self, registry: PipelineRegistry) -> None:
        class PlainCallable:
            def __call__(self) -> None:
                pass

        with pytest.raises(TypeError, match="must implement WrappedNode or a parallel protocol"):
            registry.register("bad", PlainCallable())


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


class EventD:
    pass


class ChildOfA(EventA):
    pass


class TestUnionInputExtraction:
    def test_processor_with_union_input_registers_all_types(self, registry: PipelineRegistry) -> None:
        class UnionProc(BaseThreadProcessor[EventA | EventB | EventC, EventD]):
            def __call__(self, event: EventA | EventB | EventC) -> EventD | None:
                return EventD()

        registry.register("proc", UnionProc())
        node = registry.get_node("proc")
        assert node is not None
        assert node.input_types == {EventA, EventB, EventC}
        assert node.output_types == {EventD}

    def test_sink_with_union_input_registers_all_types(self, registry: PipelineRegistry) -> None:
        class UnionSink(BaseThreadSink[EventA | EventB]):
            def __call__(self, event: EventA | EventB) -> None:
                return None

        registry.register("sink", UnionSink())
        node = registry.get_node("sink")
        assert node is not None
        assert node.input_types == {EventA, EventB}

    def test_union_input_processor_routes_from_multiple_sources(self, registry: PipelineRegistry) -> None:
        class SrcA(BaseThreadSource[EventA]):
            def __call__(self) -> EventA | None:
                return EventA()

        class SrcB(BaseThreadSource[EventB]):
            def __call__(self) -> EventB | None:
                return EventB()

        class MergeProc(BaseThreadProcessor[EventA | EventB, EventC]):
            def __call__(self, event: EventA | EventB) -> EventC | None:
                return EventC()

        registry.register("src_a", SrcA())
        registry.register("src_b", SrcB())
        registry.register("merge", MergeProc())
        registry.register("sink", EventCSink())
        errors = registry.validate()
        assert errors == []

        routing = registry.compute_routing_table()
        assert ("merge", EventA) in routing["src_a"]
        assert ("merge", EventB) in routing["src_b"]

    def test_four_sources_one_union_processor_valid(self, registry: PipelineRegistry) -> None:
        class SrcA(BaseThreadSource[EventA]):
            def __call__(self) -> EventA | None:
                return EventA()

        class SrcB(BaseThreadSource[EventB]):
            def __call__(self) -> EventB | None:
                return EventB()

        class SrcC(BaseThreadSource[EventC]):
            def __call__(self) -> EventC | None:
                return EventC()

        class SrcD(BaseThreadSource[EventD]):
            def __call__(self) -> EventD | None:
                return EventD()

        class Result:
            pass

        class MergeAll(BaseThreadProcessor[EventA | EventB | EventC | EventD, Result]):
            def __call__(self, event: EventA | EventB | EventC | EventD) -> Result | None:
                return Result()

        class ResultSink(BaseThreadSink[Result]):
            def __call__(self, event: Result) -> None:
                return None

        registry.register("src_a", SrcA())
        registry.register("src_b", SrcB())
        registry.register("src_c", SrcC())
        registry.register("src_d", SrcD())
        registry.register("merge", MergeAll())
        registry.register("sink", ResultSink())

        errors = registry.validate()
        assert errors == []

        routing = registry.compute_routing_table()
        for src_name, event_type in [("src_a", EventA), ("src_b", EventB), ("src_c", EventC), ("src_d", EventD)]:
            assert (
                "merge",
                event_type,
            ) in routing[src_name], f"{src_name} should route to merge via {event_type.__name__}"


class TestContainerTypeExtraction:
    def test_list_output_extracts_element_type(self, registry: PipelineRegistry) -> None:
        class BatchSource(BaseThreadSource[list[EventA]]):
            def __call__(self) -> list[EventA] | None:
                return [EventA()]

        registry.register("src", BatchSource())
        node = registry.get_node("src")
        assert node is not None
        assert EventA in node.output_types

    def test_tuple_ellipsis_output_extracts_element_type(self, registry: PipelineRegistry) -> None:
        class TupleSource(BaseThreadSource[tuple[EventB, ...]]):
            def __call__(self) -> tuple[EventB, ...] | None:
                return (EventB(),)

        registry.register("src", TupleSource())
        node = registry.get_node("src")
        assert node is not None
        assert EventB in node.output_types

    def test_processor_list_output_routes_to_sink(self, registry: PipelineRegistry) -> None:
        class BatchProc(BaseThreadProcessor[EventA, list[EventB]]):
            def __call__(self, event: EventA) -> list[EventB] | None:
                return [EventB()]

        registry.register("src", SimpleSource())
        registry.register("proc", BatchProc())
        registry.register("sink", SimpleSink())

        errors = registry.validate()
        assert errors == []

        routing = registry.compute_routing_table()
        assert ("sink", EventB) in routing.get("proc", [])

    def test_list_in_union_output(self, registry: PipelineRegistry) -> None:
        class MixedProc(BaseThreadProcessor[EventA, list[EventB] | EventC]):
            def __call__(self, event: EventA) -> list[EventB] | EventC | None:
                return [EventB()]

        registry.register("src", SimpleSource())
        registry.register("proc", MixedProc())
        registry.register("sink_b", SimpleSink())
        registry.register("sink_c", EventCSink())

        errors = registry.validate()
        assert errors == []

        node = registry.get_node("proc")
        assert node is not None
        assert EventB in node.output_types
        assert EventC in node.output_types


class TestInheritanceRouting:
    def test_source_declares_parent_routes_to_parent_consumer(self, registry: PipelineRegistry) -> None:
        class ParentSource(BaseThreadSource[EventA]):
            def __call__(self) -> EventA | None:
                return ChildOfA()

        class ParentSink(BaseThreadSink[EventA]):
            def __call__(self, event: EventA) -> None:
                return None

        registry.register("src", ParentSource())
        registry.register("sink", ParentSink())

        routing = registry.compute_routing_table()
        expected_types = {t for _, t in routing.get("src", [])}
        assert EventA in expected_types

    def test_subclass_source_matched_by_parent_sink(self, registry: PipelineRegistry) -> None:
        class ChildSource(BaseThreadSource[ChildOfA]):
            def __call__(self) -> ChildOfA | None:
                return ChildOfA()

        class ParentSink(BaseThreadSink[EventA]):
            def __call__(self, event: EventA) -> None:
                return None

        registry.register("src", ChildSource())
        registry.register("sink", ParentSink())

        errors = registry.validate()
        assert errors == []

    def test_subclass_routing_table_includes_parent_consumer(self, registry: PipelineRegistry) -> None:
        class ChildSource(BaseThreadSource[ChildOfA]):
            def __call__(self) -> ChildOfA | None:
                return ChildOfA()

        class ParentSink(BaseThreadSink[EventA]):
            def __call__(self, event: EventA) -> None:
                return None

        registry.register("src", ChildSource())
        registry.register("sink", ParentSink())

        routing = registry.compute_routing_table()
        assert ("sink", EventA) in routing.get("src", [])

    def test_multi_level_inheritance_routing(self, registry: PipelineRegistry) -> None:
        class GrandChildOfA(ChildOfA):
            pass

        class GrandChildSource(BaseThreadSource[GrandChildOfA]):
            def __call__(self) -> GrandChildOfA | None:
                return GrandChildOfA()

        class ParentSink(BaseThreadSink[EventA]):
            def __call__(self, event: EventA) -> None:
                return None

        registry.register("src", GrandChildSource())
        registry.register("sink", ParentSink())

        errors = registry.validate()
        assert errors == []
        routing = registry.compute_routing_table()
        assert ("sink", EventA) in routing.get("src", [])

    def test_subclass_connectivity_not_disconnected(self, registry: PipelineRegistry) -> None:
        class ChildSource(BaseThreadSource[ChildOfA]):
            def __call__(self) -> ChildOfA | None:
                return ChildOfA()

        class ParentSink(BaseThreadSink[EventA]):
            def __call__(self, event: EventA) -> None:
                return None

        registry.register("src", ChildSource())
        registry.register("sink", ParentSink())

        errors = registry.validate()
        assert not any("disconnected" in e for e in errors)


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

        class SinkA(BaseThreadSink[EventA]):
            def __call__(self, event: EventA) -> None:
                return None

        class SinkB(BaseThreadSink[EventB]):
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
        class Proc(BaseThreadProcessor[EventA, EventWithCounter]):
            def __call__(self, event: EventA) -> EventWithCounter | None:
                return EventWithCounter()

        class CounterSink(BaseThreadSink[EventWithCounter]):
            def __call__(self, event: EventWithCounter) -> None:
                pass

        registry.register("src", SimpleSource())
        registry.register("proc", Proc())
        registry.register("sink", CounterSink())
        assert "success" in registry.all_counter_names

    def test_multiple_counters_from_event_types(self, registry: PipelineRegistry) -> None:
        class ProcOk(BaseThreadProcessor[EventA, EventWithCounter | EventWithFailCounter]):
            def __call__(self, event: EventA) -> EventWithCounter | EventWithFailCounter | None:
                return EventWithCounter()

        class CounterSink(BaseThreadSink[EventWithCounter]):
            def __call__(self, event: EventWithCounter) -> None:
                pass

        class FailSink(BaseThreadSink[EventWithFailCounter]):
            def __call__(self, event: EventWithFailCounter) -> None:
                pass

        registry.register("src", SimpleSource())
        registry.register("proc", ProcOk())
        registry.register("sink_ok", CounterSink())
        registry.register("sink_fail", FailSink())
        assert "success" in registry.all_counter_names
        assert "fail" in registry.all_counter_names

    def test_annotated_does_not_break_output_types(self, registry: PipelineRegistry) -> None:
        class AnnotatedProc(BaseThreadProcessor[EventA, EventB]):
            def __call__(self, event: EventA) -> Annotated[EventB, Counter("x")] | None:
                return EventB()

        registry.register("proc", AnnotatedProc())
        node = registry.get_node("proc")
        assert node is not None
        assert EventB in node.output_types

    def test_all_counter_names(self, registry: PipelineRegistry) -> None:
        class Proc1(BaseThreadProcessor[EventA, EventWithAlpha]):
            def __call__(self, event: EventA) -> EventWithAlpha | None:
                return EventWithAlpha()

        class Proc2(BaseThreadProcessor[EventWithAlpha, EventWithBeta]):
            def __call__(self, event: EventWithAlpha) -> EventWithBeta | None:
                return EventWithBeta()

        class AlphaSink(BaseThreadSink[EventWithAlpha]):
            def __call__(self, event: EventWithAlpha) -> None:
                pass

        class BetaSink(BaseThreadSink[EventWithBeta]):
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
        from typed_flow.engine.metric_aggregator import extract_field_gauges

        gauges = extract_field_gauges(EventWithGaugeField)
        assert len(gauges) == 1
        assert gauges[0] == ("latency", Gauge("latency"))


_BASE_CLASS_VARIANTS: list[tuple[type, str]] = [
    (BaseThreadSource, "thread_source"),
    (BaseProcessSource, "process_source"),
    (BaseThreadProcessor, "thread_processor"),
    (BaseThreadPoolProcessor, "thread_pool_processor"),
    (BaseProcessPoolProcessor, "process_pool_processor"),
    (BaseThreadSink, "thread_sink"),
]

_THREADED_BASE_TO_PROTOCOL: list[tuple[type, type]] = [
    (BaseThreadSource, ThreadSource),
    (BaseProcessSource, ProcessSource),
    (BaseThreadProcessor, ThreadProcessor),
    (BaseThreadPoolProcessor, ThreadPoolProcessor),
    (BaseProcessPoolProcessor, ProcessPoolProcessor),
    (BaseThreadSink, ThreadSink),
]


class TestBaseClassVariantContract:
    @pytest.mark.parametrize(("base_cls", "expected_variant"), _BASE_CLASS_VARIANTS)
    def test_variant_is_valid_execution_variant(self, base_cls: type, expected_variant: str) -> None:
        assert expected_variant == ExecutionVariant(expected_variant).value

    @pytest.mark.parametrize(("base_cls", "expected_variant"), _BASE_CLASS_VARIANTS)
    def test_satisfies_wrapped_node(self, base_cls: type, expected_variant: str) -> None:
        instance = base_cls()
        assert isinstance(instance, WrappedNode)
        assert instance.variant == expected_variant

    @pytest.mark.parametrize(("base_cls", "protocol_cls"), _THREADED_BASE_TO_PROTOCOL)
    def test_threaded_base_satisfies_protocol(self, base_cls: type, protocol_cls: type) -> None:
        instance = base_cls()
        assert isinstance(instance, protocol_cls)

    def test_registry_detects_base_class_variant(self, registry: PipelineRegistry) -> None:
        class MyProc(BaseThreadProcessor[EventA, EventB]):
            def __call__(self, event: EventA) -> EventB | None:
                return EventB()

        registry.register("proc", MyProc())
        node = registry.get_node("proc")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_PROCESSOR

    def test_registry_detects_base_source_variant(self, registry: PipelineRegistry) -> None:
        class MySrc(BaseThreadSource[EventA]):
            def __call__(self) -> EventA | None:
                return EventA()

        registry.register("src", MySrc())
        node = registry.get_node("src")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_SOURCE

    def test_registry_detects_base_sink_variant(self, registry: PipelineRegistry) -> None:
        class MySink(BaseThreadSink[EventB]):
            def __call__(self, event: EventB) -> None:
                return None

        registry.register("sink", MySink())
        node = registry.get_node("sink")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_SINK
