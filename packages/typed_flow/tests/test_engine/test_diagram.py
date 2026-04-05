"""Tests for Mermaid diagram generation."""

from typed_flow.engine.diagram import _is_excluded_type, _node_shape, generate_mermaid
from typed_flow.engine.registry import PipelineRegistry
from typed_flow.errors.models import PipelineError
from typed_flow.metrics.collector import MetricEvent
from typed_flow.protocols import BaseThreadProcessor, BaseThreadSink, BaseThreadSource


class EventA:
    pass


class EventB:
    pass


class TestIsExcludedType:
    def test_filters_pipeline_error(self) -> None:
        assert _is_excluded_type(PipelineError) is True

    def test_filters_metric_event(self) -> None:
        assert _is_excluded_type(MetricEvent) is True

    def test_allows_regular_type(self) -> None:
        assert _is_excluded_type(EventA) is False


class TestNodeShape:
    def test_source_has_stadium_shape(self) -> None:
        result = _node_shape("src", "source")
        assert "([" in result
        assert "])" in result

    def test_sink_has_subroutine_shape(self) -> None:
        result = _node_shape("sink", "sink")
        assert "[[" in result
        assert "]]" in result

    def test_processor_has_rectangle_shape(self) -> None:
        result = _node_shape("proc", "processor")
        assert "[" in result
        assert "([" not in result
        assert "[[" not in result


class TestGenerateMermaid:
    def _build_registry(self) -> PipelineRegistry:
        class Src(BaseThreadSource[EventA]):
            def __call__(self) -> EventA | None:
                return EventA()

        class Proc(BaseThreadProcessor[EventA, EventB]):
            def __call__(self, event: EventA) -> EventB | None:
                return EventB()

        class Sink(BaseThreadSink[EventB]):
            def __call__(self, event: EventB) -> None:
                pass

        class ErrSink(BaseThreadSink[PipelineError]):
            def __call__(self, event: PipelineError) -> None:
                pass

        registry = PipelineRegistry()
        registry.register("src", Src())
        registry.register("proc", Proc())
        registry.register("sink", Sink())
        registry.register("err_sink", ErrSink())
        return registry

    def test_contains_all_node_names(self) -> None:
        diagram = generate_mermaid(self._build_registry())
        assert "src" in diagram
        assert "proc" in diagram
        assert "sink" in diagram

    def test_starts_with_graph_lr(self) -> None:
        diagram = generate_mermaid(self._build_registry())
        assert diagram.startswith("graph LR")

    def test_has_edges_with_type_labels(self) -> None:
        diagram = generate_mermaid(self._build_registry())
        assert "-->|EventA|" in diagram
        assert "-->|EventB|" in diagram

    def test_excludes_pipeline_error_edges(self) -> None:
        diagram = generate_mermaid(self._build_registry())
        assert "PipelineError" not in diagram

    def test_excludes_metric_event_edges(self) -> None:
        diagram = generate_mermaid(self._build_registry())
        assert "MetricEvent" not in diagram
