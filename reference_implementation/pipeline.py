"""CV inference pipeline: camera events → 4 models → detection/classification results.

Routes camera events by cam_id to model-specific processing branches:
  - Cams 1-2 → Model 1 (640x640 TurboJPEG detection) + Model 3 (640x640 classification)
  - Cams 3-4 → Model 2 (608x960 FP16 OpenCV detection)
  - Cams 5-6 → Model 4 (1024x1024 TurboJPEG detection with shared memory)

All tensor names, target sizes, and dtypes are auto-detected from Triton
model metadata at startup via ModelMetadataCache.

Usage:
    PYTHONPATH=. uv run python reference_implementation/pipeline.py
    PYTHONPATH=. uv run python reference_implementation/pipeline.py --config reference_implementation/config.yaml
"""

import argparse
from pathlib import Path

import tritonclient.grpc as grpcclient
import yaml
from loguru import logger
from triton_redis_connector.config.schema import DecoderType, PreprocessStrategy
from triton_redis_connector.engine.shipped_processors import (
    ClassificationPostprocessorCallable,
    DetectionPostprocessorCallable,
    InferenceNode,
    LogErrorSinkNode,
    PostProcessorNode,
    PreprocessorNode,
)
from triton_redis_connector.inference.model_metadata import ModelMetadataCache
from typed_flow.config.schema import AppConfig
from typed_flow.connectors import RedisThreadSink, RedisThreadSource
from typed_flow.engine.app import Application

from reference_implementation.models import (
    CameraEvent,
    ClassModel3Event,
    ClassModel3Output,
    ClassModel3Request,
    ClassModel3Result,
    DetModel1Event,
    DetModel1Output,
    DetModel1Request,
    DetModel1Result,
    DetModel2Event,
    DetModel2Output,
    DetModel2Request,
    DetModel2Result,
    DetModel4Event,
    DetModel4Output,
    DetModel4Request,
    DetModel4Result,
)
from reference_implementation.processors import CameraRouterProcessor

GRPC_URL = "localhost:8001"
MODEL1_NAME = "yolo11s"
MODEL2_NAME = "yolo11s_608x960"
MODEL3_NAME = "cls_resnet18"
MODEL4_NAME = "yolo11s"


class CameraSource(RedisThreadSource[CameraEvent]):
    """Reads CameraEvent from Redis stream."""


class Model1Sink(RedisThreadSink[DetModel1Result]):
    """Writes model 1 detection results to Redis stream."""


class Model2Sink(RedisThreadSink[DetModel2Result]):
    """Writes model 2 detection results to Redis stream."""


class Model4Sink(RedisThreadSink[DetModel4Result]):
    """Writes model 4 detection results to Redis stream."""


class ClassModel3Sink(RedisThreadSink[ClassModel3Result]):
    """Writes classification results to Redis stream."""


def load_config(path: str) -> AppConfig:
    """Load AppConfig from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the CV pipeline."""
    parser = argparse.ArgumentParser(description="CV inference pipeline")
    parser.add_argument("--config", default="reference_implementation/config.yaml")
    parser.add_argument("--grpc-url", default=GRPC_URL)
    return parser.parse_args()


def main() -> None:
    """Build and run the CV inference pipeline."""
    args = parse_args()
    config = load_config(args.config)
    grpc_url: str = args.grpc_url
    redis_cfg = config.redis

    source_cfg = config.sources[0]
    source = CameraSource(redis=redis_cfg, source=source_cfg, name="camera_source")

    det_sink_cfg = config.sinks[0]
    cls_sink_cfg = config.sinks[1]

    metadata_cache = ModelMetadataCache(grpcclient.InferenceServerClient(url=grpc_url))
    metadata_cache.prefetch([MODEL1_NAME, MODEL2_NAME, MODEL3_NAME, MODEL4_NAME])

    processors: list[object] = [
        source,
        CameraRouterProcessor(),
        # Preprocessors — strategy + decoder, everything else from metadata
        PreprocessorNode[DetModel1Event, DetModel1Request](
            DecoderType.TURBOJPEG,
            PreprocessStrategy.LETTERBOX,
            grpc_url,
            output_cls=DetModel1Request,
            use_shm=False,
        ),
        PreprocessorNode[DetModel2Event, DetModel2Request](
            DecoderType.OPENCV,
            PreprocessStrategy.LETTERBOX,
            grpc_url,
            output_cls=DetModel2Request,
            use_shm=False,
        ),
        PreprocessorNode[DetModel4Event, DetModel4Request](
            DecoderType.TURBOJPEG,
            PreprocessStrategy.LETTERBOX,
            grpc_url,
            output_cls=DetModel4Request,
            use_shm=True,
        ),
        PreprocessorNode[ClassModel3Event, ClassModel3Request](
            DecoderType.OPENCV,
            PreprocessStrategy.CENTER_CROP,
            grpc_url,
            output_cls=ClassModel3Request,
            use_shm=False,
        ),
        # Inference — one node per model, auto-detects tensor names
        InferenceNode[DetModel1Request, DetModel1Output](grpc_url, metadata_cache),
        InferenceNode[DetModel2Request, DetModel2Output](grpc_url, metadata_cache),
        InferenceNode[DetModel4Request, DetModel4Output](grpc_url, metadata_cache),
        InferenceNode[ClassModel3Request, ClassModel3Output](grpc_url, metadata_cache),
        # Postprocessors — composition-based typed callables
        PostProcessorNode[DetModel1Output, DetModel1Result](
            DetectionPostprocessorCallable(DetModel1Result),
        ),
        PostProcessorNode[DetModel2Output, DetModel2Result](
            DetectionPostprocessorCallable(DetModel2Result),
        ),
        PostProcessorNode[DetModel4Output, DetModel4Result](
            DetectionPostprocessorCallable(DetModel4Result),
        ),
        PostProcessorNode[ClassModel3Output, ClassModel3Result](
            ClassificationPostprocessorCallable(ClassModel3Result),
        ),
        # Per-model sinks
        Model1Sink(redis=redis_cfg, sink=det_sink_cfg, name="model1_sink"),
        Model2Sink(redis=redis_cfg, sink=det_sink_cfg, name="model2_sink"),
        Model4Sink(redis=redis_cfg, sink=det_sink_cfg, name="model4_sink"),
        ClassModel3Sink(redis=redis_cfg, sink=cls_sink_cfg, name="class_sink"),
        LogErrorSinkNode(),
    ]

    app = Application.build(config, processors=processors)

    logger.info(
        "CV pipeline: {} → [{}, {}, {}, {}] → [{}, {}]",
        source_cfg.stream_name,
        MODEL1_NAME,
        MODEL2_NAME,
        MODEL3_NAME,
        MODEL4_NAME,
        det_sink_cfg.stream_name,
        cls_sink_cfg.stream_name,
    )

    app.run_with_graceful_shutdown()


if __name__ == "__main__":
    main()
