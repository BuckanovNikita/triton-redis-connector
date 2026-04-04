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
from triton_redis_connector.engine.events import ClassificationResult, DetectionResult
from triton_redis_connector.engine.shipped_processors import LogErrorSinkNode
from typed_flow.config.schema import AppConfig
from typed_flow.connectors import RedisThreadSink, RedisThreadSource
from typed_flow.engine.app import Application
from typed_flow.errors.models import PipelineError
from typed_flow.protocols import BaseThreadSink

from reference_implementation.models import CameraEvent
from reference_implementation.processors import (
    CameraRouterProcessor,
    Model1InferenceNode,
    Model1PostProc,
    Model1Preprocessor,
    Model2InferenceNode,
    Model2PostProc,
    Model2Preprocessor,
    Model3InferenceNode,
    Model3PostProc,
    Model3Preprocessor,
    Model4InferenceNode,
    Model4PostProc,
    Model4Preprocessor,
)

GRPC_URL = "localhost:8001"
MODEL1_NAME = "yolo_640"
MODEL2_NAME = "yolo_608x960"
MODEL3_NAME = "cls_640"
MODEL4_NAME = "yolo_1024"


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
        Model1Preprocessor(model_name=MODEL1_NAME),
        Model2Preprocessor(model_name=MODEL2_NAME),
        Model3Preprocessor(model_name=MODEL3_NAME),
        Model4Preprocessor(model_name=MODEL4_NAME),
        Model1InferenceNode(grpc_url),
        Model2InferenceNode(grpc_url),
        Model3InferenceNode(grpc_url),
        Model4InferenceNode(grpc_url),
        Model1PostProc(),
        Model2PostProc(),
        Model3PostProc(),
        Model4PostProc(),
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
