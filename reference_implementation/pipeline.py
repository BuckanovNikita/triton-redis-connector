"""CV inference pipeline: camera events → 4 models → detection/classification results.

Routes camera events by cam_id to model-specific processing branches:
  - Cams 1-2 → Model 1 (640x640 TurboJPEG detection) + Model 3 (640x640 classification)
  - Cams 3-4 → Model 2 (608x960 FP16 OpenCV detection)
  - Cams 5-6 → Model 4 (1024x1024 TurboJPEG detection with shared memory)

Usage:
    PYTHONPATH=. uv run python reference_implementation/pipeline.py
    PYTHONPATH=. uv run python reference_implementation/pipeline.py --config reference_implementation/config.yaml
"""

import argparse
import signal
import sys
import types
from pathlib import Path

import yaml
from loguru import logger
from triton_redis_connector.config.schema import DecoderType
from triton_redis_connector.engine.events import ClassificationResult, DetectionResult
from triton_redis_connector.engine.shipped_processors import (
    ClassificationInferenceNode,
    ClassificationPostProcessorNode,
    ClassificationPreprocessorNode,
    ClientPreprocessorNode,
    DetectionInferenceNode,
    DetectionPostProcessorNode,
    LogErrorSinkNode,
)
from typed_flow.config.schema import AppConfig
from typed_flow.connectors import RedisThreadSink, RedisThreadSource
from typed_flow.engine.app import Application
from typed_flow.errors.models import PipelineError
from typed_flow.protocols import BaseThreadSink

from reference_implementation.models import (
    CameraEvent,
    ClassModel3Event,
    DetModel1Event,
    DetModel2Event,
    DetModel4Event,
)
from reference_implementation.processors import CameraRouterProcessor, cls_postprocess, det_postprocess

GRPC_URL = "localhost:8001"
MODEL1_NAME = "yolo11s"
MODEL2_NAME = "yolo11s_608x960"
MODEL3_NAME = "yolo12s"
MODEL4_NAME = "yolo11s"


class CameraSource(RedisThreadSource[CameraEvent]):
    """Reads CameraEvent from Redis stream."""


class DetectionSink(RedisThreadSink[DetectionResult]):
    """Writes detection results to Redis stream."""


class ClassificationSink(RedisThreadSink[ClassificationResult]):
    """Writes classification results to Redis stream."""


class ErrorSink(BaseThreadSink[PipelineError]):
    """Logs pipeline errors."""

    def __call__(self, event: PipelineError) -> None:  # noqa: D102
        logger.error("[{}] {}: {}", event.node_name, event.error_type, event.error_message)


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
    det_sink = DetectionSink(redis=redis_cfg, sink=det_sink_cfg, name="detection_sink")
    cls_sink = ClassificationSink(redis=redis_cfg, sink=cls_sink_cfg, name="classification_sink")

    processors: list[object] = [
        source,
        CameraRouterProcessor(),
        ClientPreprocessorNode[DetModel1Event](
            decoder_type=DecoderType.TURBOJPEG,
            target_size=640,
            model_name=MODEL1_NAME,
            use_shm=False,
        ),
        ClientPreprocessorNode[DetModel2Event](
            decoder_type=DecoderType.OPENCV,
            target_size=(608, 960),
            model_name=MODEL2_NAME,
            use_shm=False,
        ),
        ClientPreprocessorNode[DetModel4Event](
            decoder_type=DecoderType.TURBOJPEG,
            target_size=1024,
            model_name=MODEL4_NAME,
            use_shm=True,
        ),
        ClassificationPreprocessorNode[ClassModel3Event](
            decoder_type=DecoderType.OPENCV,
            crop_size=640,
            model_name=MODEL3_NAME,
            use_shm=False,
        ),
        DetectionInferenceNode(grpc_url),
        ClassificationInferenceNode(grpc_url),
        DetectionPostProcessorNode[DetectionResult](det_postprocess),
        ClassificationPostProcessorNode[ClassificationResult](cls_postprocess),
        det_sink,
        cls_sink,
        LogErrorSinkNode(),
        ErrorSink(),
    ]

    app = Application.build(config, processors=processors)

    shutting_down = False

    def handle_shutdown(signum: int, _frame: types.FrameType | None) -> None:
        nonlocal shutting_down
        if shutting_down:
            logger.warning("Forced exit (second {})", signal.Signals(signum).name)
            sys.exit(1)
        shutting_down = True
        logger.info("Received {}, shutting down...", signal.Signals(signum).name)
        app.shutdown()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

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

    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down...")
        app.shutdown()
    except (RuntimeError, ValueError, OSError):
        logger.exception("Pipeline crashed")
        app.shutdown()
        sys.exit(1)

    logger.info("Pipeline stopped gracefully")


if __name__ == "__main__":
    main()
