"""Tests for InferenceWorker and InferenceWorkerFactory."""

import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from triton_redis_connector.config.schema import InferenceConfig
from triton_redis_connector.inference.model_metadata import ModelMetadata, ModelMetadataCache, TensorSpec
from triton_redis_connector.inference.request import InferenceOutput, TritonRequest
from triton_redis_connector.inference.worker import InferenceWorker, InferenceWorkerFactory
from typed_flow.engine.middleware import RetryMiddleware

_FAKE_METADATA = ModelMetadata(
    name="yolo11s",
    inputs=(TensorSpec(name="images", datatype="FP32", shape=(-1, 3, 640, 640)),),
    outputs=(TensorSpec(name="output0", datatype="FP32", shape=(-1, 84, 8400)),),
)


def _make_metadata_cache() -> ModelMetadataCache:
    cache = MagicMock(spec=ModelMetadataCache)
    cache.get.return_value = _FAKE_METADATA
    return cache


def _make_input(model_name: str = "yolo11s") -> TritonRequest:
    return TritonRequest(
        model_name=model_name,
        raw_image=np.zeros((1, 3, 640, 640), dtype=np.float32),
    )


def _make_bytes_input(model_name: str = "yolo11s") -> TritonRequest:
    return TritonRequest(
        model_name=model_name,
        image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 100,
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

    class _FakeModelMetadataResponse:
        def __init__(self) -> None:
            self.name = "yolo11s"
            self.inputs = [type("T", (), {"name": "images", "datatype": "FP32", "shape": [-1, 3, 640, 640]})()]
            self.outputs = [type("T", (), {"name": "output0", "datatype": "FP32", "shape": [-1, 84, 8400]})()]

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

        def get_model_metadata(self, model_name: str, model_version: str = "") -> _FakeModelMetadataResponse:
            return _FakeModelMetadataResponse()

    module.InferInput = InferInput  # type: ignore[attr-defined]
    module.InferRequestedOutput = InferRequestedOutput  # type: ignore[attr-defined]
    module.InferenceServerClient = InferenceServerClient  # type: ignore[attr-defined]
    return module


_GRPC_PATCH_TARGET = "triton_redis_connector.inference.worker.grpcclient"


class TestInferenceWorker:
    def test_successful_inference(self) -> None:
        fake_module = _build_fake_grpc_module()
        cache = _make_metadata_cache()
        with patch(_GRPC_PATCH_TARGET, fake_module):
            worker = InferenceWorker(grpc_url="localhost:8001", metadata_cache=cache)
            result = worker(_make_input())

        assert isinstance(result, InferenceOutput)
        assert result.model_name == "yolo11s"
        assert "output0" in result.outputs
        assert result.inference_time_ms >= 0

    def test_image_bytes_inference(self) -> None:
        fake_module = _build_fake_grpc_module()
        cache = _make_metadata_cache()
        with patch(_GRPC_PATCH_TARGET, fake_module):
            worker = InferenceWorker(grpc_url="localhost:8001", metadata_cache=cache)
            result = worker(_make_bytes_input())

        assert isinstance(result, InferenceOutput)
        assert result.model_name == "yolo11s"
        assert "output0" in result.outputs
        assert result.inference_time_ms >= 0

    def test_raises_on_inference_failure(self) -> None:
        fake_module = _build_fake_grpc_module(fail_count=100)
        cache = _make_metadata_cache()
        with patch(_GRPC_PATCH_TARGET, fake_module):
            worker = InferenceWorker(grpc_url="localhost:8001", metadata_cache=cache)
            with pytest.raises(RuntimeError, match="failed"):
                worker(_make_input())

    def test_sequential_inference(self) -> None:
        fake_module = _build_fake_grpc_module()
        cache = _make_metadata_cache()
        with patch(_GRPC_PATCH_TARGET, fake_module):
            worker = InferenceWorker(grpc_url="localhost:8001", metadata_cache=cache)
            first = worker(_make_input())
            second = worker(_make_input())

        assert isinstance(first, InferenceOutput)
        assert isinstance(second, InferenceOutput)


class TestInferenceWorkerFactory:
    def test_create_shared_returns_config(self) -> None:
        config = InferenceConfig()
        factory = InferenceWorkerFactory(config=config)

        shared = factory.create_shared()

        assert shared is config

    def test_create_processor_returns_retry_wrapped_worker(self) -> None:
        config = InferenceConfig(grpc_url="test:8001")
        factory = InferenceWorkerFactory(config=config)

        fake_module = _build_fake_grpc_module()
        with patch(_GRPC_PATCH_TARGET, fake_module):
            wrapped = factory.create_processor(config)

        assert isinstance(wrapped, RetryMiddleware)

    def test_create_processor_retries_on_failure(self) -> None:
        config = InferenceConfig(
            grpc_url="test:8001",
            max_retry_attempts=3,
            retry_base_delay_s=0.001,
            retry_backoff_multiplier=1.0,
        )
        factory = InferenceWorkerFactory(config=config)

        fake_module = _build_fake_grpc_module(fail_count=2)
        with patch(_GRPC_PATCH_TARGET, fake_module):
            wrapped = factory.create_processor(config)
            result = wrapped(_make_input())

        assert isinstance(result, InferenceOutput)
        assert result.model_name == "yolo11s"
