"""Multi-task pipeline: detection (YOLO) + classification (ResNet18) via Triton.

Pipeline graph:
  ImageTaskSource
    -> TaskRouter
      -> [detect]   ClientPreprocessorNode[TaskMeta, DetectionImagePayload]
                      -> InferenceNode[TaskMeta, TritonRequest, InferenceOutput]  (shared)
                      -> PostProcessorNode (YOLO postprocessor)
                      -> DetectionResultSink -> task:detection_results
      -> [classify] ClientPreprocessorNode[TaskMeta, ClassificationImagePayload]
                      -> InferenceNode  (shared merge point)
                      -> PostProcessorNode (classification postprocessor)
                      -> ClassificationResultSink -> task:classification_results
      -> LogErrorSinkNode (InferenceError | PreprocessingError)

Two ModelBundles demonstrate multi-model routing where each branch
has its own preprocessor, postprocessor, and sink while sharing the
inference node.

Usage:
    uv run python examples/multi_task_pipeline/run_pipeline.py
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from omegaconf import OmegaConf
from redis import Redis
from triton_model_tools.inference.classification_postprocessing import ClassificationPostprocessor
from triton_model_tools.inference.yolo_bridge import YoloPostProcessorBridge
from triton_redis_connector.config.schema import (
    AppConfig,
    SourceConfig,
)
from triton_redis_connector.connectors.consumer_group import ConsumerGroupManager
from triton_redis_connector.connectors.redis_stream import RedisStreamInputConnector
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.bundle import ModelBundle, expand_bundle
from triton_redis_connector.engine.events import (
    ClassificationResult,
    DetectionResult,
    FrameworkMeta,
    TritonRequest,
)
from triton_redis_connector.engine.shipped_processors import (
    ClientPreprocessorNode,
    InferenceNode,
    LogErrorSinkNode,
    PostProcessorNode,
)
from triton_redis_connector.inference.request import InferenceOutput
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder
from triton_redis_connector.protocols import BaseThreadSource
from triton_shared_models.classification import ClassificationPrediction
from triton_shared_models.preprocessing import LetterBoxPreprocessor

from examples.multi_task_pipeline.events import (
    CLASSIFICATION_MODEL,
    CLASSIFICATION_TARGET_SIZE,
    DETECTION_MODEL,
    DETECTION_TARGET_SIZE,
    ClassificationImagePayload,
    DetectionImagePayload,
    ImageTaskInput,
    TaskMeta,
)
from examples.multi_task_pipeline.processors import (
    ClassificationResultSink,
    DetectionResultSink,
    TaskRouter,
)

if TYPE_CHECKING:
    import types

CONFIG_DIR = Path(__file__).resolve().parent

DETECTION_STREAM = "task:detection_results"
CLASSIFICATION_STREAM = "task:classification_results"


class ClassificationBridge:
    """Bridges ClassificationPostprocessor into the pipeline as a callable processor.

    Wraps raw InferenceOutput into a typed ClassificationResult.
    Inline bridge for the multi-task example (the full bridge lives
    in triton_model_tools.inference.classification_bridge).
    """

    def __init__(self, *, top_k: int = 5, apply_softmax: bool = True) -> None:
        """Initialize with top-K and softmax config."""
        self._postprocessor = ClassificationPostprocessor(top_k=top_k, apply_softmax=apply_softmax)

    def __call__(self, output: InferenceOutput) -> ClassificationResult | None:
        """Run classification postprocessing on inference output."""
        raw_predictions = self._postprocessor.postprocess(outputs=output.outputs)
        shared_predictions = tuple(
            ClassificationPrediction(
                class_id=p.class_id,
                class_name=p.class_name,
                probability=p.probability,
            )
            for p in raw_predictions
        )
        return ClassificationResult(
            predictions=shared_predictions,
            model_name=output.model_name,
            inference_time_ms=output.inference_time_ms,
        )


class ImageTaskSource(BaseThreadSource[ImageTaskInput]):
    """Reads ImageTaskInput events from a Redis stream.

    Deserializes Redis hash fields into typed domain events
    with framework metadata (source_stream, message_id).
    """

    def __init__(self, connector: RedisStreamInputConnector, config: SourceConfig) -> None:
        """Initialize source with Redis connector and config."""
        super().__init__()
        self._connector = connector
        self._config = config

    def __call__(self) -> ImageTaskInput | None:
        """Read and deserialize the next image task event from Redis."""
        envelopes = self._connector.read_new(count=self._config.batch_size, block_ms=self._config.block_ms)
        if not envelopes:
            return None
        envelope = envelopes[0]
        fields = envelope.original_fields
        return ImageTaskInput(
            fw=FrameworkMeta(
                source_stream=envelope.source_stream,
                message_id=envelope.message_id,
            ),
            task_type=fields.get("task_type", "detect"),
            image_path=fields.get("image_path", ""),
            source_id=fields.get("source_id", ""),
            priority=int(fields.get("priority", "0")),
        )


def main() -> None:
    """Load config, build multi-task pipeline with Application.build(), run."""
    raw_cfg = OmegaConf.load(CONFIG_DIR / "config.yaml")
    config = AppConfig.model_validate(OmegaConf.to_container(raw_cfg, resolve=True))
    app_name = os.environ.get("APP_NAME", "multi-task-pipeline")
    logger.info("Starting {}", app_name)

    redis_client: Redis[bytes] = Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=False,
    )

    source_cfg = config.sources[0]
    group_manager = ConsumerGroupManager()
    connector = RedisStreamInputConnector(redis_client, source_cfg, group_manager)
    source = ImageTaskSource(connector, source_cfg)

    detection_bundle = ModelBundle(
        name="detection",
        preprocessor=ClientPreprocessorNode[TaskMeta, DetectionImagePayload](
            decoder=OpenCVDecoder(),
            preprocessor=LetterBoxPreprocessor(target_size=DETECTION_TARGET_SIZE),
            model_name=DETECTION_MODEL,
        ),
        inference=InferenceNode[TaskMeta, TritonRequest, InferenceOutput](
            grpc_url=config.inference.grpc_url,
        ),
        postprocessor=PostProcessorNode[TaskMeta, InferenceOutput, DetectionResult](
            postprocessor=YoloPostProcessorBridge(),  # type: ignore[arg-type]
        ),
        sink=DetectionResultSink(redis_client, DETECTION_STREAM),
    )

    classification_bundle = ModelBundle(
        name="classification",
        preprocessor=ClientPreprocessorNode[TaskMeta, ClassificationImagePayload](
            decoder=OpenCVDecoder(),
            preprocessor=LetterBoxPreprocessor(target_size=CLASSIFICATION_TARGET_SIZE),
            model_name=CLASSIFICATION_MODEL,
        ),
        inference=InferenceNode[TaskMeta, TritonRequest, InferenceOutput](
            grpc_url=config.inference.grpc_url,
        ),
        postprocessor=PostProcessorNode[TaskMeta, InferenceOutput, ClassificationResult](
            postprocessor=ClassificationBridge(),  # type: ignore[arg-type]
        ),
        sink=ClassificationResultSink(redis_client, CLASSIFICATION_STREAM),
    )

    app = Application.build(
        config,
        processors=[
            source,
            TaskRouter(),
            *expand_bundle(detection_bundle),
            *expand_bundle(classification_bundle),
            LogErrorSinkNode(),
        ],
    )

    def handle_shutdown(signum: int, _frame: types.FrameType | None) -> None:
        logger.info("Received {}, shutting down...", signal.Signals(signum).name)
        app.shutdown()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

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
