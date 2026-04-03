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
from typed_flow.protocols.messaging import BaseEvent


class RoutedImageEvent(InferenceTarget, ImagePayload):
    model_config = ConfigDict(frozen=True)
    camera_id: str = "cam01"


class PlainImageEvent(ImagePayload):
    model_config = ConfigDict(frozen=True)
    camera_id: str = "cam01"


class PlainEvent(BaseEvent):
    model_config = ConfigDict(frozen=True, strict=True)
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


class TestAutoRedisSource:
    def test_subclass_extracts_event_model(self) -> None:
        from typed_flow.connectors.redis_stream import AutoRedisSource

        class MySource(AutoRedisSource[PlainEvent]): ...

        assert MySource._event_model is PlainEvent

    def test_subclass_without_type_param_raises(self) -> None:
        import pytest
        from typed_flow.connectors.redis_stream import AutoRedisSource

        with pytest.raises(TypeError, match="must specify a concrete BaseModel type"):

            class BadSource(AutoRedisSource): ...  # type: ignore[type-arg]

    def test_produces_deserialized_event(self) -> None:
        from unittest.mock import MagicMock

        from typed_flow.config.schema import SourceConfig
        from typed_flow.connectors.redis_stream import (
            AutoRedisSource,
            RedisStreamInputConnector,
        )
        from typed_flow.protocols import Envelope

        class MySource(AutoRedisSource[PlainEvent]): ...

        connector = MagicMock(spec=RedisStreamInputConnector)
        config = SourceConfig(
            stream_name="stream:test",
            consumer_group="workers",
            consumer_name="c1",
            event_model="PlainEvent",
        )

        source = MySource(connector, config, name="test_auto")

        envelope = Envelope(
            payload=b"",
            source_stream="stream:test",
            message_id="1-0",
            original_fields={"camera_id": "cam99"},
        )
        connector.read_new.return_value = [envelope]

        result = source()
        assert isinstance(result, PlainEvent)
        assert result.camera_id == "cam99"
        assert result.flow_meta.source_stream == "stream:test"

    def test_returns_none_when_no_messages(self) -> None:
        from unittest.mock import MagicMock

        from typed_flow.config.schema import SourceConfig
        from typed_flow.connectors.redis_stream import (
            AutoRedisSource,
            RedisStreamInputConnector,
        )

        class MySource(AutoRedisSource[PlainEvent]): ...

        connector = MagicMock(spec=RedisStreamInputConnector)
        config = SourceConfig(
            stream_name="stream:test",
            consumer_group="workers",
            consumer_name="c1",
            event_model="PlainEvent",
        )
        source = MySource(connector, config)

        connector.read_new.return_value = []
        result = source()
        assert result is None
