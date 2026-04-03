"""Dual-model steel inspection pipeline: Redis Stream -> Branch -> Triton -> Redis Stream.

Pipeline graph:
  SteelInspectionSource
    -> CameraBranchRouter
      -> [even] ClientPreprocessorNode[SteelMeta, EvenImagePayload]
      -> [odd]  ClientPreprocessorNode[SteelMeta, OddImagePayload]
            -> InferenceNode  (shared merge point)
            -> PostProcessorNode  (shared merge point)
                -> ModelResultRouter
                  -> [even] EvenResultSink -> steel:even_results
                  -> [odd]  OddResultSink  -> steel:odd_results
               -> LogErrorSinkNode (InferenceError | PreprocessingError)

Usage:
    uv run python examples/dual_model_pipeline/run_pipeline.py
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
from triton_model_tools.inference.yolo_bridge import YoloPostProcessorBridge
from triton_redis_connector.config.schema import (
    AppConfig,
    SourceConfig,
)
from triton_redis_connector.connectors.consumer_group import ConsumerGroupManager
from triton_redis_connector.connectors.redis_stream import RedisStreamInputConnector
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.events import (
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
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder, TurboJPEGDecoder
from triton_redis_connector.protocols import BaseThreadSource
from triton_shared_models.preprocessing import LetterBoxPreprocessor

from examples.dual_model_pipeline.events import (
    EVEN_MODEL,
    ODD_MODEL,
    EvenImagePayload,
    OddImagePayload,
    SteelInspectionInput,
    SteelMeta,
)
from examples.dual_model_pipeline.processors import (
    CameraBranchRouter,
    EvenResultSink,
    ModelResultRouter,
    OddResultSink,
)

if TYPE_CHECKING:
    import types

CONFIG_DIR = Path(__file__).resolve().parent

EVEN_STREAM = "steel:even_results"
ODD_STREAM = "steel:odd_results"

EVEN_TARGET_SIZE = (608, 960)
ODD_TARGET_SIZE = (640, 640)


class SteelInspectionSource(BaseThreadSource[SteelInspectionInput]):
    """Reads SteelInspectionInput events from a Redis stream.

    Wraps the low-level RedisStreamInputConnector to produce typed
    domain events with framework metadata (source_stream, message_id).
    """

    def __init__(self, connector: RedisStreamInputConnector, config: SourceConfig) -> None:
        """Initialize source with Redis connector and config."""
        super().__init__()
        self._connector = connector
        self._config = config

    def __call__(self) -> SteelInspectionInput | None:
        """Read and deserialize the next steel inspection event from Redis."""
        envelopes = self._connector.read_new(count=self._config.batch_size, block_ms=self._config.block_ms)
        if not envelopes:
            return None
        envelope = envelopes[0]
        fields = envelope.original_fields
        return SteelInspectionInput(
            fw=FrameworkMeta(
                source_stream=envelope.source_stream,
                message_id=envelope.message_id,
            ),
            receipt_ts=float(fields.get("receipt_ts", "0")),
            camera_no=int(fields.get("camera_no", "0")),
            image_path=fields.get("image_path", ""),
            frame_counter=int(fields.get("frame_counter", "0")),
            metal_type=fields.get("metal_type"),
        )


def main() -> None:
    """Load config, build pipeline with Application.build(), run."""
    raw_cfg = OmegaConf.load(CONFIG_DIR / "config.yaml")
    config = AppConfig.model_validate(OmegaConf.to_container(raw_cfg, resolve=True))
    app_name = os.environ.get("APP_NAME", "dual-model-steel-inspection")
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
    source = SteelInspectionSource(connector, source_cfg)

    app = Application.build(
        config,
        processors=[
            source,
            CameraBranchRouter(),
            ClientPreprocessorNode[SteelMeta, EvenImagePayload](
                decoder=OpenCVDecoder(),
                preprocessor=LetterBoxPreprocessor(target_size=EVEN_TARGET_SIZE),
                model_name=EVEN_MODEL,
            ),
            ClientPreprocessorNode[SteelMeta, OddImagePayload](
                decoder=TurboJPEGDecoder(),
                preprocessor=LetterBoxPreprocessor(target_size=ODD_TARGET_SIZE),
                model_name=ODD_MODEL,
            ),
            InferenceNode[SteelMeta, TritonRequest, InferenceOutput](
                grpc_url=config.inference.grpc_url,
            ),
            PostProcessorNode[SteelMeta, InferenceOutput, DetectionResult](
                postprocessor=YoloPostProcessorBridge(  # type: ignore[arg-type]
                    conf_threshold=config.pipeline.postprocessor.conf_threshold,
                    iou_threshold=config.pipeline.postprocessor.iou_threshold,
                    end2end=config.pipeline.postprocessor.end2end,
                ),
            ),
            ModelResultRouter(),
            EvenResultSink(redis_client, EVEN_STREAM),
            OddResultSink(redis_client, ODD_STREAM),
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
