"""Tests for PreprocessorNode with model_name from ImagePayload."""

from unittest.mock import MagicMock

import numpy as np
from pydantic import ConfigDict
from triton_redis_connector.config.schema import DecoderType, PreprocessStrategy
from triton_redis_connector.engine.events import (
    ImagePayload,
    TritonRequest,
)
from triton_redis_connector.engine.shipped_processors import PreprocessorNode
from triton_redis_connector.inference.model_metadata import ModelMetadata, TensorSpec


class PlainImageEvent(ImagePayload):
    model_config = ConfigDict(frozen=True)
    camera_id: str = "cam01"


class PlainTritonRequest(TritonRequest):
    """Marker for test model requests."""


_FAKE_METADATA = ModelMetadata(
    name="yolo11s",
    inputs=(TensorSpec(name="images", datatype="FP32", shape=(-1, 3, 640, 640)),),
    outputs=(TensorSpec(name="output0", datatype="FP32", shape=(-1, 84, 8400)),),
)


def _make_node(model_name: str = "yolo11s") -> PreprocessorNode[PlainImageEvent, PlainTritonRequest]:
    node: PreprocessorNode[PlainImageEvent, PlainTritonRequest] = PreprocessorNode(
        decoder_type=DecoderType.OPENCV,
        strategy=PreprocessStrategy.LETTERBOX,
        grpc_url="localhost:8001",
        output_cls=PlainTritonRequest,
        num_workers=1,
    )
    mock_decoder = MagicMock()
    mock_decoder.decode.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
    mock_preprocessor = MagicMock()
    mock_preprocessor.preprocess.return_value = (
        np.zeros((1, 3, 640, 640), dtype=np.float32),
        1.0,
        (0, 0),
    )
    mock_metadata_cache = MagicMock()
    mock_metadata_cache.get.return_value = _FAKE_METADATA
    node._decoder = mock_decoder  # pyright: ignore[reportPrivateUsage]
    node._preprocessor = mock_preprocessor  # pyright: ignore[reportPrivateUsage]
    node._metadata_cache = mock_metadata_cache  # pyright: ignore[reportPrivateUsage]
    node._model_metadata = {model_name: _FAKE_METADATA}  # pyright: ignore[reportPrivateUsage]
    return node


class TestPreprocessorNodeRouting:
    def test_uses_model_name_from_event(self) -> None:
        node = _make_node(model_name="yolo11s")
        event = PlainImageEvent(model_name="yolo11s", image_bytes=b"fake-jpeg")

        result = node(event)

        assert isinstance(result, PlainTritonRequest)
        assert result.model_name == "yolo11s"

    def test_preserves_preprocess_scale_and_pad(self) -> None:
        node = _make_node(model_name="yolo11s")
        mock_preprocessor = MagicMock()
        mock_preprocessor.preprocess.return_value = (
            np.zeros((1, 3, 640, 640), dtype=np.float32),
            0.75,
            (32, 16),
        )
        node._preprocessor = mock_preprocessor  # pyright: ignore[reportPrivateUsage]

        event = PlainImageEvent(model_name="yolo11s", image_bytes=b"fake-jpeg")
        result = node(event)

        assert isinstance(result, PlainTritonRequest)
        assert result.preprocess_scale == 0.75
        assert result.preprocess_pad == (32, 16)

    def test_returns_correct_output_type(self) -> None:
        node = _make_node(model_name="yolo11s")
        event = PlainImageEvent(model_name="yolo11s", image_bytes=b"fake-jpeg")

        result = node(event)

        assert type(result) is PlainTritonRequest
