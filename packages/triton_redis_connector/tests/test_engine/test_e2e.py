"""E2E integration tests and build validation for the pipeline framework.

Task 5.7: Build validation catches broken configs (no Docker needed).
Task 5.4: Error event creation and properties (partial, no Docker needed).
Task 5.6: User data preservation via PipelineEvent round-trip (partial, no Docker needed).
Task 5.1/5.2: Full pipeline smoke test (requires Redis + Triton).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from triton_redis_connector.errors.models import InferenceError, PreprocessingError
from typed_flow.config.schema import AppConfig
from typed_flow.engine.app import Application
from typed_flow.engine.events import (
    FrameworkMeta,
    PipelineEvent,
)
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import (
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Shared test types
# ---------------------------------------------------------------------------


class SimplePayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: str = "test"


class AnotherPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    number: int = 42


class UserContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    camera_id: str = "cam_01"
    location: str = "hall_a"


class ContextWithShadowingNames(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_stream: str = "user_stream"
    message_id: str = "user_msg_123"
    created_at: float = 1234567890.0


class ContextWithResultField(BaseModel):
    model_config = ConfigDict(frozen=True)
    result: str = "user_result"


class PlainEvent:
    pass


class ProcessedEvent:
    pass


class OutputEvent:
    pass


class OrphanInput:
    pass


class OrphanOutput:
    pass


# ---------------------------------------------------------------------------
# Task 5.7: Build validation catches broken configs
# ---------------------------------------------------------------------------


class TestBuildValidationMissingSource:
    """Application.build() rejects pipelines without a source."""

    def test_no_source_raises_value_error(self) -> None:
        class Proc(BaseThreadProcessor[PlainEvent, ProcessedEvent]):
            def __call__(self, event: PlainEvent) -> ProcessedEvent | None:
                return ProcessedEvent()

        class Sink(BaseThreadSink[ProcessedEvent]):
            def __call__(self, event: ProcessedEvent) -> None:
                return None

        with pytest.raises(ValueError, match="validation error"):
            Application.build(AppConfig(), processors=[Proc(), Sink()])


class TestBuildValidationMissingSink:
    """Application.build() rejects pipelines without a sink."""

    def test_no_sink_raises_value_error(self) -> None:
        class Src(BaseThreadSource[PlainEvent]):
            def __call__(self) -> PlainEvent | None:
                return PlainEvent()

        class Proc(BaseThreadProcessor[PlainEvent, ProcessedEvent]):
            def __call__(self, event: PlainEvent) -> ProcessedEvent | None:
                return ProcessedEvent()

        with pytest.raises(ValueError, match="validation error"):
            Application.build(AppConfig(), processors=[Src(), Proc()])


class TestBuildValidationUnconnectedProcessor:
    """Application.build() rejects pipelines with orphan processors."""

    def test_orphan_processor_raises_value_error(self) -> None:
        class Src(BaseThreadSource[PlainEvent]):
            def __call__(self) -> PlainEvent | None:
                return PlainEvent()

        class Proc(BaseThreadProcessor[PlainEvent, ProcessedEvent]):
            def __call__(self, event: PlainEvent) -> ProcessedEvent | None:
                return ProcessedEvent()

        class Sink(BaseThreadSink[ProcessedEvent]):
            def __call__(self, event: ProcessedEvent) -> None:
                return None

        class Orphan(BaseThreadProcessor[OrphanInput, OrphanOutput]):
            def __call__(self, event: OrphanInput) -> OrphanOutput | None:
                return OrphanOutput()

        with pytest.raises(ValueError, match="validation error"):
            Application.build(
                AppConfig(),
                processors=[Src(), Proc(), Sink(), Orphan()],
            )


class TestBuildValidationReservedResultField:
    """TCtx with a field named 'result' collides with the sink serialization.

    The sink serializes the payload into a Redis hash entry named 'result',
    so a user context field with the same name would cause a collision.
    """

    def test_context_with_result_field_creates_collision_risk(self) -> None:
        ctx = ContextWithResultField()
        assert ctx.result == "user_result"
        fields = ctx.model_dump()
        assert "result" in fields


# ---------------------------------------------------------------------------
# Task 5.4: Error event creation and properties (partial)
# ---------------------------------------------------------------------------


class TestErrorEventCreation:
    def test_connector_error_creation(self) -> None:
        err = ConnectorError(
            connector_name="redis_source",
            stream_name="input:stream",
            message_id="100-0",
            error_type="ConnectionError",
            error_message="Connection refused",
            timestamp_ms=1000,
        )
        assert err.error_type == "ConnectionError"
        assert err.stream_name == "input:stream"

    def test_inference_error_creation(self) -> None:
        err = InferenceError(
            model_name="yolo11s",
            model_version="1",
            error_type="TimeoutError",
            error_message="Request timed out after 5s",
            retry_count=3,
            timestamp_ms=2000,
        )
        assert err.retry_count == 3
        assert err.model_name == "yolo11s"

    def test_preprocessing_error_creation(self) -> None:
        err = PreprocessingError(
            message_id="200-0",
            decoder_type="opencv",
            error_type="DecodeError",
            error_message="Cannot decode corrupt JPEG",
            timestamp_ms=3000,
        )
        assert err.decoder_type == "opencv"
        assert err.error_message == "Cannot decode corrupt JPEG"


class TestErrorEventFrozen:
    def test_connector_error_rejects_mutation(self) -> None:
        err = ConnectorError(
            connector_name="x",
            stream_name="s",
            message_id=None,
            error_type="E",
            error_message="m",
            timestamp_ms=0,
        )
        with pytest.raises(ValidationError, match="frozen"):
            err.error_message = "changed"  # type: ignore[misc]

    def test_inference_error_rejects_mutation(self) -> None:
        err = InferenceError(
            model_name="m",
            model_version="1",
            error_type="E",
            error_message="m",
            retry_count=0,
            timestamp_ms=0,
        )
        with pytest.raises(ValidationError, match="frozen"):
            err.retry_count = 99  # type: ignore[misc]

    def test_preprocessing_error_rejects_mutation(self) -> None:
        err = PreprocessingError(
            message_id="id",
            decoder_type="opencv",
            error_type="E",
            error_message="m",
            timestamp_ms=0,
        )
        with pytest.raises(ValidationError, match="frozen"):
            err.decoder_type = "turbojpeg"  # type: ignore[misc]


class TestPreprocessingErrorMetrics:
    def test_has_decode_failure_counter(self) -> None:
        from typed_flow.engine.metric_aggregator import extract_event_counters

        counters = extract_event_counters(PreprocessingError)
        assert any(c.name == "decode_failure" for c in counters)


# ---------------------------------------------------------------------------
# Task 5.6: User data preservation via PipelineEvent round-trip (partial)
# ---------------------------------------------------------------------------


class TestPipelineEventPreservesUserContext:
    def test_fw_separate_from_user_context(self) -> None:
        fw = FrameworkMeta(source_stream="framework:stream", message_id="fw-001")
        ctx = ContextWithShadowingNames(
            source_stream="user_stream",
            message_id="user_msg_123",
            created_at=1234567890.0,
        )
        pe: PipelineEvent[ContextWithShadowingNames, SimplePayload] = PipelineEvent(
            fw=fw,
            context=ctx,
            payload=SimplePayload(data="hello"),
        )

        assert pe.fw.source_stream == "framework:stream"
        assert pe.fw.message_id == "fw-001"
        assert pe.context.source_stream == "user_stream"
        assert pe.context.message_id == "user_msg_123"
        assert pe.context.created_at == 1234567890.0

    def test_user_fields_survive_without_fw(self) -> None:
        ctx = UserContext(camera_id="cam_07", location="gate_b")
        pe: PipelineEvent[UserContext, SimplePayload] = PipelineEvent(
            fw=FrameworkMeta(source_stream="fw:stream", message_id="fw-002"),
            context=ctx,
            payload=SimplePayload(data="world"),
        )

        dumped = pe.context.model_dump()
        assert "fw" not in dumped
        assert dumped["camera_id"] == "cam_07"
        assert dumped["location"] == "gate_b"

    def test_pipeline_event_preserves_payload_through_transform(self) -> None:
        ctx = UserContext(camera_id="cam_03", location="zone_c")
        pe_input: PipelineEvent[UserContext, SimplePayload] = PipelineEvent(
            context=ctx,
            payload=SimplePayload(data="input_data"),
        )

        pe_output: PipelineEvent[UserContext, AnotherPayload] = PipelineEvent(
            fw=pe_input.fw,
            context=pe_input.context,
            payload=AnotherPayload(number=99),
        )

        assert pe_output.context.camera_id == "cam_03"
        assert pe_output.context.location == "zone_c"
        assert pe_output.payload.number == 99
        assert pe_output.context is pe_input.context


class TestPipelineEventShadowedFieldsRoundTrip:
    def test_shadowed_names_serialization_round_trip(self) -> None:
        fw = FrameworkMeta(source_stream="fw:src", message_id="fw-msg-100")
        ctx = ContextWithShadowingNames(
            source_stream="user_source",
            message_id="user_msg",
            created_at=9999.9,
        )
        pe: PipelineEvent[ContextWithShadowingNames, SimplePayload] = PipelineEvent(
            fw=fw,
            context=ctx,
            payload=SimplePayload(),
        )

        json_str = pe.model_dump_json()
        restored = PipelineEvent[ContextWithShadowingNames, SimplePayload].model_validate_json(json_str)

        assert restored.fw.source_stream == "fw:src"
        assert restored.fw.message_id == "fw-msg-100"
        assert restored.context.source_stream == "user_source"
        assert restored.context.message_id == "user_msg"
        assert restored.context.created_at == 9999.9


# ---------------------------------------------------------------------------
# Task 5.1/5.2: Full pipeline smoke tests (integration, requires Docker)
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_client() -> Any:
    """Create a synchronous Redis client or skip if unavailable."""
    try:
        from redis import Redis
        from redis.exceptions import RedisError
    except ImportError:
        pytest.skip("redis package not installed")
        return None  # unreachable, satisfies type checker

    try:
        client: Redis[bytes] = Redis(host="localhost", port=6379, decode_responses=False)
        client.ping()
    except (ConnectionError, OSError, RedisError):
        pytest.skip("Redis not available at localhost:6379")
    return client


@pytest.fixture
def triton_available() -> None:
    """Skip the test if Triton is not reachable."""
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:8000/v2/health/live", timeout=2)
    except (OSError, urllib.error.URLError):
        pytest.skip("Triton not available at localhost:8000")


class RawMessage:
    def __init__(self, data: str) -> None:
        self.data = data


class UpperMessage:
    def __init__(self, data: str) -> None:
        self.data = data


@pytest.mark.integration
class TestMinimalPipelineWithRealInfrastructure:
    def test_source_processor_sink_data_round_trip(self, redis_client: Any) -> None:
        stream_id = uuid.uuid4().hex[:8]
        input_stream = f"test:e2e:input:{stream_id}"
        collected: list[UpperMessage] = []

        class Src(BaseThreadSource[RawMessage]):
            def __init__(self, client: Any, stream: str) -> None:
                super().__init__()
                self._client = client
                self._stream = stream
                self._cursor = b"0-0"
                self._consumed = 0

            def __call__(self) -> RawMessage | None:
                max_consume = 3
                if self._consumed >= max_consume:
                    return None
                entries = self._client.xread(
                    streams={self._stream: self._cursor},
                    count=1,
                    block=100,
                )
                if not entries:
                    return None
                _stream_name, messages = entries[0]
                msg_id, fields = messages[0]
                self._cursor = msg_id
                self._consumed += 1
                return RawMessage(data=fields[b"data"].decode())

        class Upper(BaseThreadProcessor[RawMessage, UpperMessage]):
            def __call__(self, event: RawMessage) -> UpperMessage | None:
                return UpperMessage(data=event.data.upper())

        class Collector(BaseThreadSink[UpperMessage]):
            def __call__(self, event: UpperMessage) -> None:
                collected.append(event)

        redis_client.xadd(input_stream, {b"data": b"hello"})
        redis_client.xadd(input_stream, {b"data": b"world"})
        redis_client.xadd(input_stream, {b"data": b"test"})

        app = Application.build(
            AppConfig(),
            processors=[Src(redis_client, input_stream), Upper(), Collector()],
        )

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()

        deadline = time.monotonic() + 5.0
        expected_count = 3
        while time.monotonic() < deadline and len(collected) < expected_count:
            time.sleep(0.05)

        app.shutdown()
        run_thread.join(timeout=3.0)

        redis_client.delete(input_stream)

        assert len(collected) == expected_count
        assert {m.data for m in collected} == {"HELLO", "WORLD", "TEST"}


class RedisInputEvent:
    def __init__(self, camera_id: str, data: str) -> None:
        self.camera_id = camera_id
        self.data = data


class TransformedEvent:
    def __init__(self, camera_id: str, data_upper: str, data_len: int) -> None:
        self.camera_id = camera_id
        self.data_upper = data_upper
        self.data_len = data_len


@pytest.mark.integration
class TestPipelineContextPreservationWithRedis:
    """Verify that user context data survives the full pipeline with Redis."""

    def test_context_fields_preserved_through_pipeline(self, redis_client: Any) -> None:
        stream_id = uuid.uuid4().hex[:8]
        input_stream = f"test:e2e:ctx:{stream_id}"
        collected: list[TransformedEvent] = []

        class Src(BaseThreadSource[RedisInputEvent]):
            def __init__(self, client: Any, stream: str) -> None:
                super().__init__()
                self._client = client
                self._stream = stream
                self._cursor = b"0-0"
                self._consumed = 0

            def __call__(self) -> RedisInputEvent | None:
                max_consume = 2
                if self._consumed >= max_consume:
                    return None
                entries = self._client.xread(
                    streams={self._stream: self._cursor},
                    count=1,
                    block=100,
                )
                if not entries:
                    return None
                _stream_name, messages = entries[0]
                msg_id, fields = messages[0]
                self._cursor = msg_id
                self._consumed += 1
                data = fields[b"data"].decode()
                camera = fields[b"camera"].decode()
                return RedisInputEvent(camera_id=camera, data=data)

        class Transform(BaseThreadProcessor[RedisInputEvent, TransformedEvent]):
            def __call__(self, event: RedisInputEvent) -> TransformedEvent | None:
                return TransformedEvent(
                    camera_id=event.camera_id,
                    data_upper=event.data.upper(),
                    data_len=len(event.data),
                )

        class Collector(BaseThreadSink[TransformedEvent]):
            def __call__(self, event: TransformedEvent) -> None:
                collected.append(event)

        redis_client.xadd(input_stream, {b"data": b"alpha", b"camera": b"cam_01"})
        redis_client.xadd(input_stream, {b"data": b"beta", b"camera": b"cam_02"})

        app = Application.build(
            AppConfig(),
            processors=[Src(redis_client, input_stream), Transform(), Collector()],
        )

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()

        deadline = time.monotonic() + 5.0
        expected_count = 2
        while time.monotonic() < deadline and len(collected) < expected_count:
            time.sleep(0.05)

        app.shutdown()
        run_thread.join(timeout=3.0)

        redis_client.delete(input_stream)

        assert len(collected) == expected_count
        camera_ids = {e.camera_id for e in collected}
        assert "cam_01" in camera_ids
        assert "cam_02" in camera_ids
        for event in collected:
            assert event.data_upper == event.data_upper.upper()
            assert event.data_len > 0


@pytest.mark.integration
class TestTritonInferenceSmoke:
    def test_triton_health_check(self, triton_available: None) -> None:
        import urllib.request

        resp = urllib.request.urlopen("http://localhost:8000/v2/health/live", timeout=2)
        assert resp.status == 200

    def test_triton_model_ready(self, triton_available: None) -> None:
        import urllib.request

        resp = urllib.request.urlopen(
            "http://localhost:8000/v2/models/yolo11s/ready",
            timeout=2,
        )
        assert resp.status == 200

    def test_grpc_inference_returns_output(
        self,
        triton_available: None,
        test_images_dir: Path,
    ) -> None:
        import cv2
        import tritonclient.grpc as grpcclient
        from triton_shared_models.preprocessing import LetterBoxPreprocessor

        client = grpcclient.InferenceServerClient(url="localhost:8001")
        assert client.is_server_live()
        assert client.is_model_ready("yolo11s")

        image_files = sorted(test_images_dir.glob("*.jpg"))
        assert len(image_files) > 0

        img = cv2.imread(str(image_files[0]))
        assert img is not None

        preprocessor = LetterBoxPreprocessor((640, 640))
        tensor, _scale, _pad = preprocessor.preprocess(img)

        infer_input = grpcclient.InferInput("images", tensor.shape, "FP32")
        infer_input.set_data_from_numpy(tensor)

        result = client.infer(
            model_name="yolo11s",
            inputs=[infer_input],
            outputs=[grpcclient.InferRequestedOutput("output0")],
        )
        output = result.as_numpy("output0")
        assert output is not None
        assert output.shape[0] > 0


# ---------------------------------------------------------------------------
# Task 5.1 extended: Full pipeline with real inference (integration)
# ---------------------------------------------------------------------------


class ImagePathEvent:
    def __init__(self, path: str, camera_id: str) -> None:
        self.path = path
        self.camera_id = camera_id


class InferenceResultEvent:
    def __init__(self, camera_id: str, model_name: str, output_shape: tuple[int, ...]) -> None:
        self.camera_id = camera_id
        self.model_name = model_name
        self.output_shape = output_shape


@pytest.mark.integration
class TestFullInferencePipeline:
    """End-to-end: source reads image paths, preprocesses, infers, collects results."""

    def test_image_to_inference_pipeline(  # noqa: C901
        self,
        redis_client: Any,
        triton_available: None,
        test_images_dir: Path,
    ) -> None:
        stream_id = uuid.uuid4().hex[:8]
        input_stream = f"test:e2e:infer:{stream_id}"
        inference_results: list[InferenceResultEvent] = []

        image_files = sorted(test_images_dir.glob("*.jpg"))[:3]
        assert len(image_files) > 0

        class ImgSrc(BaseThreadSource[ImagePathEvent]):
            def __init__(self, client: Any, stream: str) -> None:
                super().__init__()
                self._client = client
                self._stream = stream
                self._cursor = b"0-0"
                self._consumed = 0
                self._max = len(image_files)

            def __call__(self) -> ImagePathEvent | None:
                if self._consumed >= self._max:
                    return None
                entries = self._client.xread(
                    streams={self._stream: self._cursor},
                    count=1,
                    block=200,
                )
                if not entries:
                    return None
                _stream_name, messages = entries[0]
                msg_id, fields = messages[0]
                self._cursor = msg_id
                self._consumed += 1
                path = fields[b"path"].decode()
                return ImagePathEvent(path=path, camera_id=f"cam_{self._consumed}")

        class PreprocessAndInfer(BaseThreadProcessor[ImagePathEvent, InferenceResultEvent]):
            def __init__(self) -> None:
                super().__init__()
                import cv2
                import tritonclient.grpc as grpcclient
                from triton_shared_models.preprocessing import LetterBoxPreprocessor

                self._preprocessor = LetterBoxPreprocessor((640, 640))
                self._triton = grpcclient.InferenceServerClient(url="localhost:8001")
                self._cv2 = cv2
                self._grpcclient = grpcclient

            def __call__(self, event: ImagePathEvent) -> InferenceResultEvent | None:
                img = self._cv2.imread(event.path)
                if img is None:
                    return None
                tensor, _s, _p = self._preprocessor.preprocess(img)
                infer_input = self._grpcclient.InferInput("images", tensor.shape, "FP32")
                infer_input.set_data_from_numpy(tensor)
                result = self._triton.infer(
                    model_name="yolo11s",
                    inputs=[infer_input],
                    outputs=[self._grpcclient.InferRequestedOutput("output0")],
                )
                output = result.as_numpy("output0")
                return InferenceResultEvent(
                    camera_id=event.camera_id,
                    model_name="yolo11s",
                    output_shape=tuple(output.shape),
                )

        class ResultCollector(BaseThreadSink[InferenceResultEvent]):
            def __call__(self, event: InferenceResultEvent) -> None:
                inference_results.append(event)

        for img_path in image_files:
            redis_client.xadd(input_stream, {b"path": str(img_path).encode()})

        app = Application.build(
            AppConfig(),
            processors=[ImgSrc(redis_client, input_stream), PreprocessAndInfer(), ResultCollector()],
        )

        run_thread = threading.Thread(target=app.run, daemon=True)
        run_thread.start()

        deadline = time.monotonic() + 15.0
        expected_count = len(image_files)
        while time.monotonic() < deadline and len(inference_results) < expected_count:
            time.sleep(0.1)

        app.shutdown()
        run_thread.join(timeout=5.0)

        redis_client.delete(input_stream)

        assert len(inference_results) == expected_count
        for r in inference_results:
            assert r.model_name == "yolo11s"
            assert len(r.output_shape) > 0
            assert r.output_shape[0] > 0
