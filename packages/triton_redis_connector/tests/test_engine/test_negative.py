"""Negative tests — error handling, validation rejection, type mismatches.

Verifies that the pipeline framework correctly rejects invalid configurations,
handles processor failures gracefully, and enforces exact type matching.
"""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import BaseModel, ConfigDict
from triton_redis_connector.config.schema import AppConfig
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.events import (
    FrameworkMeta,
    PipelineEvent,
)
from triton_redis_connector.engine.registry import PipelineRegistry
from triton_redis_connector.engine.validation import validate_graph
from triton_redis_connector.errors.models import ConnectorError, InferenceError, PreprocessingError
from triton_redis_connector.protocols import (
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
)


class PayloadA(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: str = "a"


class PayloadB(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: str = "b"


class MyCtx(BaseModel):
    model_config = ConfigDict(frozen=True)
    camera_id: str = "cam_1"


class EventX:
    pass


class EventY:
    pass


class EventZ:
    pass


class TestBuildValidationRejectsInvalidConfigs:
    """Application.build() must reject pipelines missing source or sink."""

    def test_build_with_no_source_raises(self) -> None:
        class Proc(BaseThreadProcessor[EventX, EventY]):
            def __call__(self, event: EventX) -> EventY | None:
                return EventY()

        class Sink(BaseThreadSink[EventY]):
            def __call__(self, event: EventY) -> None:
                return None

        with pytest.raises(ValueError, match="validation error"):
            Application.build(AppConfig(), processors=[Proc(), Sink()])

    def test_build_with_no_sink_raises(self) -> None:
        class Src(BaseThreadSource[EventX]):
            def __call__(self) -> EventX | None:
                return EventX()

        class Proc(BaseThreadProcessor[EventX, EventY]):
            def __call__(self, event: EventX) -> EventY | None:
                return EventY()

        with pytest.raises(ValueError, match="validation error"):
            Application.build(AppConfig(), processors=[Src(), Proc()])

    def test_build_with_orphan_processor_raises(self) -> None:
        class Src(BaseThreadSource[EventX]):
            def __call__(self) -> EventX | None:
                return EventX()

        class Proc(BaseThreadProcessor[EventX, EventY]):
            def __call__(self, event: EventX) -> EventY | None:
                return EventY()

        class Sink(BaseThreadSink[EventY]):
            def __call__(self, event: EventY) -> None:
                return None

        class OrphanProc(BaseThreadProcessor[EventZ, EventZ]):
            def __call__(self, event: EventZ) -> EventZ | None:
                return EventZ()

        with pytest.raises(ValueError, match="validation error"):
            Application.build(
                AppConfig(),
                processors=[Src(), Proc(), Sink(), OrphanProc()],
            )


class TestTypeMismatchDetection:
    """Register a processor producing type A with only a consumer for type B."""

    def test_unconsumed_output_detected(self) -> None:
        class Src(BaseThreadSource[EventX]):
            def __call__(self) -> EventX | None:
                return EventX()

        class Proc(BaseThreadProcessor[EventX, EventY]):
            def __call__(self, event: EventX) -> EventY | None:
                return EventY()

        class WrongSink(BaseThreadSink[EventZ]):
            def __call__(self, event: EventZ) -> None:
                return None

        registry = PipelineRegistry()
        registry.register("src", Src())
        registry.register("proc", Proc())
        registry.register("sink", WrongSink())

        errors = validate_graph(registry)
        assert len(errors) > 0
        error_text = " ".join(errors)
        assert "EventY" in error_text or "no node consumes" in error_text


class TestPipelineEventExactMatching:
    """PipelineEvent[CtxA, PayloadA] must NOT match PipelineEvent[CtxA, PayloadB]."""

    def test_pipeline_event_different_payload_not_routed(self) -> None:
        class Src(BaseThreadSource[PipelineEvent[MyCtx, PayloadA]]):
            def __call__(self) -> PipelineEvent[MyCtx, PayloadA] | None:
                return PipelineEvent(context=MyCtx(), payload=PayloadA())

        class SinkB(BaseThreadSink[PipelineEvent[MyCtx, PayloadB]]):
            def __call__(self, event: PipelineEvent[MyCtx, PayloadB]) -> None:
                return None

        registry = PipelineRegistry()
        registry.register("src", Src())
        registry.register("sink_b", SinkB())

        errors = registry.validate()
        assert len(errors) > 0
        error_text = " ".join(errors)
        assert "no node consumes" in error_text or "no node produces" in error_text

    def test_pipeline_event_same_payload_matches(self) -> None:
        class Src(BaseThreadSource[PipelineEvent[MyCtx, PayloadA]]):
            def __call__(self) -> PipelineEvent[MyCtx, PayloadA] | None:
                return PipelineEvent(context=MyCtx(), payload=PayloadA())

        class SinkA(BaseThreadSink[PipelineEvent[MyCtx, PayloadA]]):
            def __call__(self, event: PipelineEvent[MyCtx, PayloadA]) -> None:
                return None

        registry = PipelineRegistry()
        registry.register("src", Src())
        registry.register("sink_a", SinkA())

        errors = registry.validate()
        assert errors == []


class TestProcessorResilienceOnException:
    """A processor that raises on the 3rd call must not kill the node thread."""

    def test_thread_survives_processor_exception(self) -> None:
        call_count = 0
        results: list[EventY] = []

        class Src(BaseThreadSource[EventX]):
            def __init__(self) -> None:
                super().__init__()
                self._count = 0

            def __call__(self) -> EventX | None:
                self._count += 1
                max_events = 5
                if self._count <= max_events:
                    return EventX()
                return None

        class FailingProc(BaseThreadProcessor[EventX, EventY]):
            def __call__(self, event: EventX) -> EventY | None:
                nonlocal call_count
                call_count += 1
                fail_on_call = 3
                if call_count == fail_on_call:
                    msg = "intentional failure on 3rd call"
                    raise ValueError(msg)
                return EventY()

        class CollectSink(BaseThreadSink[EventY]):
            def __call__(self, event: EventY) -> None:
                results.append(event)

        app = Application.build(
            AppConfig(),
            processors=[Src(), FailingProc(), CollectSink()],
        )

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()

        deadline = time.monotonic() + 3.0
        expected_results = 4
        while time.monotonic() < deadline and len(results) < expected_results:
            time.sleep(0.05)

        app.shutdown()
        run_thread.join(timeout=2.0)

        assert call_count >= 3
        assert len(results) >= 2


class TestErrorEventFields:
    """All error event types have the required fields populated."""

    def test_connector_error_has_all_fields(self) -> None:
        err = ConnectorError(
            connector_name="redis_source",
            stream_name="input:stream",
            message_id="123-0",
            error_type="ConnectionError",
            error_message="Connection refused",
            timestamp_ms=1000,
        )
        assert err.connector_name == "redis_source"
        assert err.stream_name == "input:stream"
        assert err.message_id == "123-0"
        assert err.error_type == "ConnectionError"
        assert err.error_message == "Connection refused"
        assert err.timestamp_ms == 1000

    def test_inference_error_has_all_fields(self) -> None:
        err = InferenceError(
            model_name="yolo11s",
            model_version="1",
            error_type="TimeoutError",
            error_message="Request timed out",
            retry_count=3,
            timestamp_ms=2000,
        )
        assert err.model_name == "yolo11s"
        assert err.model_version == "1"
        assert err.error_type == "TimeoutError"
        assert err.error_message == "Request timed out"
        assert err.retry_count == 3
        assert err.timestamp_ms == 2000

    def test_preprocessing_error_has_all_fields(self) -> None:
        err = PreprocessingError(
            message_id="456-0",
            decoder_type="opencv",
            error_type="DecodeError",
            error_message="Cannot decode image",
            timestamp_ms=3000,
        )
        assert err.message_id == "456-0"
        assert err.decoder_type == "opencv"
        assert err.error_type == "DecodeError"
        assert err.error_message == "Cannot decode image"
        assert err.timestamp_ms == 3000

    def test_connector_error_frozen(self) -> None:
        from pydantic import ValidationError

        err = ConnectorError(
            connector_name="x",
            stream_name="s",
            message_id=None,
            error_type="E",
            error_message="m",
            timestamp_ms=0,
        )
        with pytest.raises(ValidationError, match="frozen"):
            err.connector_name = "y"  # type: ignore[misc]

    def test_inference_error_frozen(self) -> None:
        from pydantic import ValidationError

        err = InferenceError(
            model_name="m",
            model_version="1",
            error_type="E",
            error_message="m",
            retry_count=0,
            timestamp_ms=0,
        )
        with pytest.raises(ValidationError, match="frozen"):
            err.model_name = "other"  # type: ignore[misc]


class TestValidateGraphStructural:
    """validate_graph rejects structurally broken pipelines."""

    def test_empty_registry_reports_no_nodes(self) -> None:
        registry = PipelineRegistry()
        errors = validate_graph(registry)
        assert any("No nodes" in e for e in errors)

    def test_source_only_reports_missing_sink(self) -> None:
        class Src(BaseThreadSource[EventX]):
            def __call__(self) -> EventX | None:
                return EventX()

        registry = PipelineRegistry()
        registry.register("src", Src())
        errors = validate_graph(registry)
        assert any("sink" in e.lower() for e in errors)

    def test_sink_only_reports_missing_source(self) -> None:
        class Sink(BaseThreadSink[EventX]):
            def __call__(self, event: EventX) -> None:
                return None

        registry = PipelineRegistry()
        registry.register("sink", Sink())
        errors = validate_graph(registry)
        assert any("source" in e.lower() or "No source" in e for e in errors)


class TestBuildDeduplicatesNodeNames:
    """Application.build appends _N suffix when node names collide."""

    def test_duplicate_class_names_get_suffix(self) -> None:
        class Src(BaseThreadSource[EventX]):
            def __call__(self) -> EventX | None:
                return EventX()

        class Proc(BaseThreadProcessor[EventX, EventY]):
            def __call__(self, event: EventX) -> EventY | None:
                return EventY()

        class Sink(BaseThreadSink[EventY]):
            def __call__(self, event: EventY) -> None:
                return None

        app = Application.build(
            AppConfig(),
            processors=[Src(), Proc(), Proc(), Sink()],
        )
        node_names = list(app._registry.nodes.keys())  # pyright: ignore[reportPrivateUsage]
        assert "Proc" in node_names
        assert "Proc_1" in node_names


class TestRoutingTableExactMatch:
    """Routing table uses exact type identity for dispatch."""

    def test_routing_table_maps_producer_to_consumers(self) -> None:
        class Src(BaseThreadSource[EventX]):
            def __call__(self) -> EventX | None:
                return EventX()

        class Proc(BaseThreadProcessor[EventX, EventY]):
            def __call__(self, event: EventX) -> EventY | None:
                return EventY()

        class Sink(BaseThreadSink[EventY]):
            def __call__(self, event: EventY) -> None:
                return None

        registry = PipelineRegistry()
        registry.register("src", Src())
        registry.register("proc", Proc())
        registry.register("sink", Sink())

        table = registry.compute_routing_table()
        assert "src" in table
        assert any(name == "proc" for name, _ in table["src"])
        assert "proc" in table
        assert any(name == "sink" for name, _ in table["proc"])

    def test_union_output_routes_to_separate_consumers(self) -> None:
        class Src(BaseThreadSource[EventX]):
            def __call__(self) -> EventX | None:
                return EventX()

        class Router(BaseThreadProcessor[EventX, EventY | EventZ]):
            def __call__(self, event: EventX) -> EventY | EventZ | None:
                return EventY()

        class SinkY(BaseThreadSink[EventY]):
            def __call__(self, event: EventY) -> None:
                return None

        class SinkZ(BaseThreadSink[EventZ]):
            def __call__(self, event: EventZ) -> None:
                return None

        registry = PipelineRegistry()
        registry.register("src", Src())
        registry.register("router", Router())
        registry.register("sink_y", SinkY())
        registry.register("sink_z", SinkZ())

        table = registry.compute_routing_table()
        router_consumers = table["router"]
        consumer_names = {name for name, _ in router_consumers}
        assert "sink_y" in consumer_names
        assert "sink_z" in consumer_names


class TestFrameworkMetaPreserved:
    """FrameworkMeta travels through PipelineEvent.fw."""

    def test_fw_accessible_through_pipeline_event(self) -> None:
        fw = FrameworkMeta(source_stream="stream:input", message_id="100-0", priority=1)
        ctx = MyCtx(camera_id="cam_1")
        pe: PipelineEvent[MyCtx, PayloadA] = PipelineEvent(fw=fw, context=ctx, payload=PayloadA())
        assert pe.fw.source_stream == "stream:input"
        assert pe.fw.message_id == "100-0"
        assert pe.fw.priority == 1
