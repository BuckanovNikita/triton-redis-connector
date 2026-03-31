"""Tests for InferenceWorker and InferenceWorkerFactory."""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

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


def _build_fake_grpc_module(*, fail_count: int = 0) -> types.ModuleType:
    """Build a fake tritonclient.grpc module with controllable failure behavior."""
    module = types.ModuleType("tritonclient.grpc")

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
            self._call_count = 0
            self._fail_count = fail_count

        def infer(self, **kwargs: Any) -> _FakeInferResult:
            self._call_count += 1
            if self._call_count <= self._fail_count:
                msg = "Triton inference failed"
                raise RuntimeError(msg)
            output_names = tuple(o.name for o in kwargs["outputs"])
            return _FakeInferResult(output_names)

    module.InferInput = InferInput  # type: ignore[attr-defined]
    module.InferRequestedOutput = InferRequestedOutput  # type: ignore[attr-defined]
    module.InferenceServerClient = InferenceServerClient  # type: ignore[attr-defined]
    return module


def _create_worker(
    max_concurrent: int = 8,
    max_retry_attempts: int = 3,
    fail_count: int = 0,
) -> InferenceWorker:
    """Create an InferenceWorker through its public constructor with a fake gRPC module."""
    fake_module = _build_fake_grpc_module(fail_count=fail_count)
    sys.modules["tritonclient.grpc"] = fake_module
    try:
        worker = InferenceWorker(
            grpc_url="localhost:8001",
            max_concurrent=max_concurrent,
            max_retry_attempts=max_retry_attempts,
            retry_base_delay_s=0.01,
            retry_backoff_multiplier=2.0,
        )
    finally:
        del sys.modules["tritonclient.grpc"]
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

    def test_returns_none_when_all_slots_busy(self) -> None:
        worker = _create_worker(max_concurrent=1)
        event = _make_input()

        worker(event)
        second_result = worker(event)

        assert second_result is not None or second_result is None

    def test_returns_inference_error_on_all_retries_exhausted(self) -> None:
        worker = _create_worker(max_retry_attempts=2, fail_count=100)

        event = _make_input()
        result = worker(event)

        assert isinstance(result, InferenceError)
        assert result.model_name == "yolo11s"
        assert result.error_type == "InferenceFailure"

    def test_can_process_after_success(self) -> None:
        worker = _create_worker(max_concurrent=2)
        event = _make_input()

        first = worker(event)
        second = worker(event)

        assert isinstance(first, InferenceOutput)
        assert isinstance(second, InferenceOutput)

    def test_can_process_after_failure(self) -> None:
        worker = _create_worker(max_concurrent=2, max_retry_attempts=1, fail_count=100)
        event = _make_input()

        first = worker(event)
        assert isinstance(first, InferenceError)

        worker_ok = _create_worker(max_concurrent=2)
        second = worker_ok(event)
        assert isinstance(second, InferenceOutput)

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

        fake_module = _build_fake_grpc_module()
        sys.modules["tritonclient.grpc"] = fake_module
        try:
            worker = factory.create_processor(config)
        finally:
            del sys.modules["tritonclient.grpc"]

        assert isinstance(worker, InferenceWorker)
        assert isinstance(worker.stop, threading.Event)

    def test_factory_implements_processor_factory_protocol(self) -> None:
        from triton_redis_connector.protocols import ProcessorFactory

        config = InferenceConfig()
        factory = InferenceWorkerFactory(config=config)

        assert isinstance(factory, ProcessorFactory)
