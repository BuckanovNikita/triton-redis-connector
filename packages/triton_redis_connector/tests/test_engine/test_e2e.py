"""E2E integration tests for the pipeline framework.

Task 5.1/5.2: Full pipeline smoke test (requires Redis + Triton).
"""

import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from triton_redis_connector.errors.models import PreprocessingError
from typed_flow.config.schema import AppConfig
from typed_flow.engine.app import Application
from typed_flow.protocols import (
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
)


class TestPreprocessingErrorMetrics:
    def test_has_decode_failure_counter(self) -> None:
        from typed_flow.engine.metric_aggregator import extract_event_counters

        counters = extract_event_counters(PreprocessingError)
        assert any(c.name == "decode_failure" for c in counters)


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
        from triton_shared_models.detection.preprocessing import LetterBoxPreprocessor

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
                from triton_shared_models.detection.preprocessing import LetterBoxPreprocessor

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
