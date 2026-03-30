"""Tests for Triton model configuration and YOLO model registry."""

from __future__ import annotations

from triton_model_tools.models import (
    EXPORT_FORMAT_TO_FILENAME,
    EXPORT_FORMAT_TO_PLATFORM,
    YOLO_MODELS,
    DynamicBatchingConfig,
    ExportFormat,
    ModelWarmupConfig,
    OptimizationConfig,
    TritonDataType,
    TritonInstanceKind,
    TritonModelConfig,
    TritonTensorSpec,
    WarmupInputSpec,
    YoloModelVariant,
)


class TestTritonTensorSpec:
    def test_creates_tensor_spec(self) -> None:
        spec = TritonTensorSpec(
            name="images",
            data_type=TritonDataType.TYPE_FP32,
            dims=[1, 3, 640, 640],
        )
        assert spec.name == "images"
        assert spec.data_type == TritonDataType.TYPE_FP32
        assert spec.dims == [1, 3, 640, 640]

    def test_is_shape_tensor_default(self) -> None:
        spec = TritonTensorSpec(
            name="data",
            data_type=TritonDataType.TYPE_FP32,
            dims=[1],
        )
        assert spec.is_shape_tensor is False


class TestTritonModelConfig:
    def test_to_pbtxt_renders_valid_config(self) -> None:
        config = TritonModelConfig(
            name="test_model",
            inputs=[
                TritonTensorSpec(
                    name="images",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 3, 640, 640],
                ),
            ],
            outputs=[
                TritonTensorSpec(
                    name="output0",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 300, 6],
                ),
            ],
        )
        pbtxt = config.to_pbtxt()

        assert 'name: "test_model"' in pbtxt
        assert 'platform: "onnxruntime_onnx"' in pbtxt
        assert "max_batch_size: 0" in pbtxt
        assert 'name: "images"' in pbtxt
        assert "data_type: TYPE_FP32" in pbtxt
        assert "dims: [ 1, 3, 640, 640 ]" in pbtxt
        assert 'name: "output0"' in pbtxt
        assert "dims: [ 1, 300, 6 ]" in pbtxt
        assert "kind: KIND_GPU" in pbtxt
        assert "count: 1" in pbtxt

    def test_to_pbtxt_with_cpu_instance(self) -> None:
        config = TritonModelConfig(
            name="cpu_model",
            instance_kind=TritonInstanceKind.KIND_CPU,
            instance_count=2,
            inputs=[
                TritonTensorSpec(
                    name="input",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[-1],
                ),
            ],
            outputs=[
                TritonTensorSpec(
                    name="output",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[-1],
                ),
            ],
        )
        pbtxt = config.to_pbtxt()

        assert "kind: KIND_CPU" in pbtxt
        assert "count: 2" in pbtxt

    def test_to_pbtxt_multiple_outputs(self) -> None:
        config = TritonModelConfig(
            name="multi_out",
            inputs=[
                TritonTensorSpec(
                    name="images",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 3, 640, 640],
                ),
            ],
            outputs=[
                TritonTensorSpec(
                    name="output0",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 84, 8400],
                ),
                TritonTensorSpec(
                    name="output1",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 32, 160, 160],
                ),
            ],
        )
        pbtxt = config.to_pbtxt()

        assert pbtxt.count("output [") == 2
        assert 'name: "output0"' in pbtxt
        assert 'name: "output1"' in pbtxt

    def test_defaults(self) -> None:
        config = TritonModelConfig(
            name="defaults_test",
            inputs=[
                TritonTensorSpec(name="x", data_type=TritonDataType.TYPE_FP32, dims=[1]),
            ],
            outputs=[
                TritonTensorSpec(name="y", data_type=TritonDataType.TYPE_FP32, dims=[1]),
            ],
        )
        assert config.platform == "onnxruntime_onnx"
        assert config.max_batch_size == 0
        assert config.instance_count == 1
        assert config.instance_kind == TritonInstanceKind.KIND_GPU
        assert config.default_model_filename == ""
        assert config.dynamic_batching is None
        assert config.optimization is None
        assert config.model_warmup is None


class TestExportFormatReexports:
    def test_export_format_accessible(self) -> None:
        assert ExportFormat.ONNX == "onnx"
        assert ExportFormat.TENSORRT == "tensorrt"

    def test_platform_mapping_accessible(self) -> None:
        assert EXPORT_FORMAT_TO_PLATFORM[ExportFormat.ONNX] == "onnxruntime_onnx"

    def test_filename_mapping_accessible(self) -> None:
        assert EXPORT_FORMAT_TO_FILENAME[ExportFormat.ONNX] == "model.onnx"


class TestNewConfigModelsReexports:
    def test_dynamic_batching_config(self) -> None:
        config = DynamicBatchingConfig(
            max_queue_delay_microseconds=100,
            preferred_batch_size=[1, 4],
        )
        assert config.max_queue_delay_microseconds == 100

    def test_optimization_config(self) -> None:
        config = OptimizationConfig(graph_level=2, cuda_graphs=True)
        assert config.graph_level == 2

    def test_warmup_input_spec(self) -> None:
        spec = WarmupInputSpec(
            data_type=TritonDataType.TYPE_FP32,
            dims=[3, 640, 640],
        )
        assert spec.zero_data is True

    def test_model_warmup_config(self) -> None:
        warmup = ModelWarmupConfig(
            name="warmup",
            batch_size=1,
            inputs={
                "images": WarmupInputSpec(
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[3, 640, 640],
                ),
            },
        )
        assert warmup.batch_size == 1


class TestYoloModelVariant:
    def test_creates_variant_with_defaults(self) -> None:
        variant = YoloModelVariant(
            name="test",
            ultralytics_model="test.pt",
        )
        assert variant.export_kwargs == {}
        assert variant.imgsz == 640
        assert variant.end2end_export is False

    def test_creates_variant_with_kwargs(self) -> None:
        variant = YoloModelVariant(
            name="yolo26s",
            ultralytics_model="yolo26s.pt",
            export_kwargs={"end2end": True},
        )
        assert variant.export_kwargs == {"end2end": True}

    def test_creates_variant_with_end2end_export(self) -> None:
        variant = YoloModelVariant(
            name="yolo26s",
            ultralytics_model="yolo26s.pt",
            end2end_export=True,
        )
        assert variant.end2end_export is True


class TestYoloModelsRegistry:
    def test_registry_has_three_models(self) -> None:
        assert len(YOLO_MODELS) == 3

    def test_registry_model_names(self) -> None:
        names = {v.name for v in YOLO_MODELS}
        assert names == {"yolo26s", "yolo11s", "yolo12s"}

    def test_yolo26s_has_end2end(self) -> None:
        yolo26s = next(v for v in YOLO_MODELS if v.name == "yolo26s")
        assert yolo26s.export_kwargs.get("end2end") is True

    def test_yolo26s_has_end2end_export_true(self) -> None:
        yolo26s = next(v for v in YOLO_MODELS if v.name == "yolo26s")
        assert yolo26s.end2end_export is True

    def test_non_end2end_models_have_end2end_export_false(self) -> None:
        non_e2e = [v for v in YOLO_MODELS if v.name != "yolo26s"]
        assert len(non_e2e) > 0
        for variant in non_e2e:
            assert variant.end2end_export is False

    def test_all_models_use_pt_extension(self) -> None:
        for variant in YOLO_MODELS:
            assert variant.ultralytics_model.endswith(".pt")

    def test_all_models_default_imgsz(self) -> None:
        for variant in YOLO_MODELS:
            assert variant.imgsz == 640
