"""Tests for ModelMetadataCache."""

from unittest.mock import MagicMock

from triton_redis_connector.inference.model_metadata import (
    ModelMetadata,
    ModelMetadataCache,
    TensorSpec,
    _parse_metadata_response,
)


class _FakeTensorMetadata:
    def __init__(self, name: str, datatype: str, shape: list[int]) -> None:
        self.name = name
        self.datatype = datatype
        self.shape = shape


class _FakeMetadataResponse:
    def __init__(
        self,
        name: str,
        inputs: list[_FakeTensorMetadata],
        outputs: list[_FakeTensorMetadata],
    ) -> None:
        self.name = name
        self.inputs = inputs
        self.outputs = outputs


class TestModelMetadata:
    def test_input_name(self) -> None:
        meta = ModelMetadata(
            name="yolo11s",
            inputs=(TensorSpec(name="images", datatype="FP32", shape=(-1, 3, 640, 640)),),
            outputs=(TensorSpec(name="output0", datatype="FP32", shape=(-1, 84, 8400)),),
        )
        assert meta.input_name == "images"

    def test_output_names(self) -> None:
        meta = ModelMetadata(
            name="model",
            inputs=(TensorSpec(name="input", datatype="FP32", shape=(1, 3, 640, 640)),),
            outputs=(
                TensorSpec(name="out1", datatype="FP32", shape=(1, 84, 8400)),
                TensorSpec(name="out2", datatype="FP32", shape=(1, 10)),
            ),
        )
        assert meta.output_names == ("out1", "out2")

    def test_target_size_from_nchw(self) -> None:
        meta = ModelMetadata(
            name="model",
            inputs=(TensorSpec(name="images", datatype="FP32", shape=(-1, 3, 640, 640)),),
            outputs=(TensorSpec(name="output0", datatype="FP32", shape=(-1, 84, 8400)),),
        )
        assert meta.target_size == (640, 640)

    def test_target_size_non_square(self) -> None:
        meta = ModelMetadata(
            name="model",
            inputs=(TensorSpec(name="images", datatype="FP16", shape=(-1, 3, 608, 960)),),
            outputs=(TensorSpec(name="output0", datatype="FP32", shape=(-1, 84, 8400)),),
        )
        assert meta.target_size == (608, 960)

    def test_input_dtype_str(self) -> None:
        meta = ModelMetadata(
            name="model",
            inputs=(TensorSpec(name="images", datatype="FP16", shape=(-1, 3, 640, 640)),),
            outputs=(TensorSpec(name="output0", datatype="FP32", shape=(-1, 84, 8400)),),
        )
        assert meta.input_dtype_str == "FP16"


class TestParseMetadataResponse:
    def test_parses_protobuf_response(self) -> None:
        response = _FakeMetadataResponse(
            name="yolo11s",
            inputs=[_FakeTensorMetadata("images", "FP32", [-1, 3, 640, 640])],
            outputs=[_FakeTensorMetadata("output0", "FP32", [-1, 84, 8400])],
        )
        meta = _parse_metadata_response(response)
        assert meta.name == "yolo11s"
        assert meta.input_name == "images"
        assert meta.output_names == ("output0",)
        assert meta.target_size == (640, 640)


class TestModelMetadataCache:
    def test_get_fetches_and_caches(self) -> None:
        response = _FakeMetadataResponse(
            name="yolo11s",
            inputs=[_FakeTensorMetadata("images", "FP32", [-1, 3, 640, 640])],
            outputs=[_FakeTensorMetadata("output0", "FP32", [-1, 84, 8400])],
        )
        mock_client = MagicMock()
        mock_client.get_model_metadata.return_value = response

        cache = ModelMetadataCache(mock_client)
        first = cache.get("yolo11s")
        second = cache.get("yolo11s")

        assert first is second
        mock_client.get_model_metadata.assert_called_once()

    def test_prefetch_caches_multiple_models(self) -> None:
        responses = {
            ("yolo11s", ""): _FakeMetadataResponse(
                "yolo11s",
                [_FakeTensorMetadata("images", "FP32", [-1, 3, 640, 640])],
                [_FakeTensorMetadata("output0", "FP32", [-1, 84, 8400])],
            ),
            ("resnet18", ""): _FakeMetadataResponse(
                "resnet18",
                [_FakeTensorMetadata("input", "FP32", [-1, 3, 224, 224])],
                [_FakeTensorMetadata("output", "FP32", [-1, 1000])],
            ),
        }
        mock_client = MagicMock()
        mock_client.get_model_metadata.side_effect = lambda name, version="": responses[(name, version)]

        cache = ModelMetadataCache(mock_client)
        cache.prefetch(["yolo11s", "resnet18"])

        assert cache.get("yolo11s").name == "yolo11s"
        assert cache.get("resnet18").name == "resnet18"
        assert mock_client.get_model_metadata.call_count == 2
