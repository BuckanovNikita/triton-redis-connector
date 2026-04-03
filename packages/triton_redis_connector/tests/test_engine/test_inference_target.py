"""Tests for InferenceTarget dynamic model routing in ClientPreprocessorNode."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from pydantic import ConfigDict
from triton_redis_connector.engine.events import (
    ImagePayload,
    InferenceTarget,
    TritonRequest,
)
from triton_redis_connector.engine.shipped_processors import ClientPreprocessorNode


class RoutedImageEvent(InferenceTarget, ImagePayload):
    model_config = ConfigDict(frozen=True)
    camera_id: str = "cam01"


class PlainImageEvent(ImagePayload):
    model_config = ConfigDict(frozen=True)
    camera_id: str = "cam01"


def _make_node(model_name: str = "yolo11s") -> ClientPreprocessorNode[RoutedImageEvent]:
    decoder = MagicMock()
    decoder.decode.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

    preprocessor = MagicMock()
    preprocessor.preprocess.return_value = (
        np.zeros((1, 3, 640, 640), dtype=np.float32),
        1.0,
        (0, 0),
    )

    return ClientPreprocessorNode(
        decoder=decoder,
        preprocessor=preprocessor,
        model_name=model_name,
        input_name="images",
        output_names=("output0",),
    )


class TestInferenceTargetRouting:
    def test_uses_target_model_from_event(self) -> None:
        node = _make_node(model_name="yolo11s")
        event = RoutedImageEvent(target_model="yolo26l", image_bytes=b"fake-jpeg")

        result = node(event)

        assert isinstance(result, TritonRequest)
        assert result.model_name == "yolo26l"

    def test_uses_target_model_version_from_event(self) -> None:
        node = _make_node(model_name="yolo11s")
        event = RoutedImageEvent(
            target_model="yolo26l",
            target_model_version="2",
            image_bytes=b"fake-jpeg",
        )

        result = node(event)

        assert isinstance(result, TritonRequest)
        assert result.model_name == "yolo26l"
        assert result.model_version == "2"

    def test_falls_back_to_default_model_without_inference_target(self) -> None:
        node: ClientPreprocessorNode[PlainImageEvent] = ClientPreprocessorNode(
            decoder=MagicMock(decode=MagicMock(return_value=np.zeros((640, 640, 3), dtype=np.uint8))),
            preprocessor=MagicMock(
                preprocess=MagicMock(
                    return_value=(np.zeros((1, 3, 640, 640), dtype=np.float32), 1.0, (0, 0)),
                ),
            ),
            model_name="yolo11s",
            input_name="images",
            output_names=("output0",),
        )
        event = PlainImageEvent(image_bytes=b"fake-jpeg")

        result = node(event)

        assert isinstance(result, TritonRequest)
        assert result.model_name == "yolo11s"
        assert result.model_version == ""

    def test_falls_back_when_target_model_is_empty(self) -> None:
        node = _make_node(model_name="yolo11s")
        event = RoutedImageEvent(target_model="", image_bytes=b"fake-jpeg")

        result = node(event)

        assert isinstance(result, TritonRequest)
        assert result.model_name == "yolo11s"
