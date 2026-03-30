"""Tests for InferenceWorker and InferenceWorkerFactory."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from triton_redis_connector.config.schema import InferenceConfig
from triton_redis_connector.errors.models import InferenceError
from triton_redis_connector.inference.request import InferenceInput, InferenceOutput
from triton_redis_connector.inference.worker import InferenceWorker, InferenceWorkerFactory


def _make_input(model_name: str = "yolo11s") -> InferenceInput:
    return InferenceInput(
        model_name=model_name,
        raw_image=np.zeros((1, 3, 640, 640), dtype=np.float32),
    )


class _FakeInferResult:
    def __init__(self, output_names: tuple[str, ...]) -> None:
        self._data = {name: np.zeros((1, 84, 8400), dtype=np.float32) for name in output_names}

    def as_numpy(self, name: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        return self._data[name]


class _FakeGrpcModule:
    class InferInput:
        def __init__(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
            self.name = name

        def set_data_from_numpy(self, data: Any) -> None:
            pass

    class InferRequestedOutput:
        def __init__(self, name: str) -> None:
            self.name = name

    class InferenceServerClient:
        def __init__(self, url: str = "") -> None:
            self.url = url
            self._result: _FakeInferResult | None = None
            self._should_fail = False
            self._fail_count = 0
            self._call_count = 0

        def infer(self, **kwargs: Any) -> _FakeInferResult:
            self._call_count += 1
            if self._should_fail and self._call_count <= self._fail_count:
                msg = "Triton inference failed"
                raise RuntimeError(msg)
            output_names = tuple(o.name for o in kwargs["outputs"])
            return _FakeInferResult(output_names)


def _noop_init(*_args: Any, **_kwargs: Any) -> None:
    pass


def _create_worker(
    max_concurrent: int = 8,
    max_retry_attempts: int = 3,
) -> InferenceWorker:
    with patch("triton_redis_connector.inference.worker.InferenceWorker.__init__", _noop_init):
        worker = object.__new__(InferenceWorker)

    fake_grpc = _FakeGrpcModule()
    client = _FakeGrpcModule.InferenceServerClient(url="localhost:8001")

    worker._grpc_module = fake_grpc
    worker._client = client
    worker._semaphore = threading.Semaphore(max_concurrent)
    worker.stop = threading.Event()

    from triton_redis_connector.inference.retry import RetryHandler

    worker._retry_handler = RetryHandler(
        max_attempts=max_retry_attempts,
        base_delay_s=0.01,
        backoff_multiplier=2.0,
    )
    return worker


class TestInferenceWorker:
    def test_successful_inference(self) -> None:
        worker = _create_worker()
        event = _make_input()

        result = worker(event)

        assert isinstance(result, InferenceOutput)
        assert result.model_name == "yolo11s"
        assert "output0" in result.outputs
        assert result.inference_time_ms > 0

    def test_returns_none_when_semaphore_full(self) -> None:
        worker = _create_worker(max_concurrent=1)
        worker._semaphore.acquire()

        event = _make_input()
        result = worker(event)

        assert result is None
        worker._semaphore.release()

    def test_returns_inference_error_on_all_retries_exhausted(self) -> None:
        worker = _create_worker(max_retry_attempts=2)
        worker._client._should_fail = True
        worker._client._fail_count = 100

        event = _make_input()
        result = worker(event)

        assert isinstance(result, InferenceError)
        assert result.model_name == "yolo11s"
        assert result.error_type == "InferenceFailure"

    def test_semaphore_released_on_success(self) -> None:
        worker = _create_worker(max_concurrent=2)
        event = _make_input()

        worker(event)

        assert worker._semaphore._value == 2

    def test_semaphore_released_on_failure(self) -> None:
        worker = _create_worker(max_concurrent=2, max_retry_attempts=1)
        worker._client._should_fail = True
        worker._client._fail_count = 100

        event = _make_input()
        worker(event)

        assert worker._semaphore._value == 2

    def test_stop_event_exists(self) -> None:
        worker = _create_worker()
        assert isinstance(worker.stop, threading.Event)
        assert not worker.stop.is_set()


class TestInferenceWorkerFactory:
    def test_create_shared_returns_config(self) -> None:
        config = InferenceConfig()
        factory = InferenceWorkerFactory(config=config)

        shared = factory.create_shared()

        assert shared is config

    def test_create_processor_returns_worker(self) -> None:
        config = InferenceConfig(grpc_url="test:8001", max_concurrent_per_worker=4)
        factory = InferenceWorkerFactory(config=config)

        with patch(
            "tritonclient.grpc.InferenceServerClient",
            return_value=MagicMock(),
        ):
            worker = factory.create_processor(config)

        assert isinstance(worker, InferenceWorker)
        assert isinstance(worker.stop, threading.Event)

    def test_factory_implements_processor_factory_protocol(self) -> None:
        from triton_redis_connector.protocols import ProcessorFactory

        config = InferenceConfig()
        factory = InferenceWorkerFactory(config=config)

        assert isinstance(factory, ProcessorFactory)
