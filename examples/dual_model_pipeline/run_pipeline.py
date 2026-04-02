"""Dual-model steel inspection pipeline: Redis Stream -> Branch -> Triton -> Redis Stream.

Pipeline graph:
  RedisStreamSource -> EnvelopeExtractor[SteelInspectionInput]
    -> CameraBranchRouter
      -> [even] ClientPreprocessor --+-> InferenceWorker
      -> [odd]  ClientPreprocessor --+     -> YoloPostProcessorBridge
                                                -> ModelResultRouter
                                                  -> [even] EvenModelSink -> steel:even_results
                                                  -> [odd]  OddModelSink  -> steel:odd_results
                                           -> LogErrorSink (InferenceError)

Usage:
    uv run python examples/dual_model_pipeline/run_pipeline.py
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

from loguru import logger
from redis import Redis
from triton_model_tools.inference.yolo_bridge import YoloPostProcessorBridge
from triton_redis_connector.config.hydra_app import load_config
from triton_redis_connector.providers.app_factory import build_application

from examples.dual_model_pipeline.events import SteelInspectionInput
from examples.dual_model_pipeline.processors import (
    CameraBranchRouter,
    EvenModelSink,
    ModelResultRouter,
    OddModelSink,
)

CONFIG_DIR = Path(__file__).resolve().parent

EVEN_STREAM = "steel:even_results"
ODD_STREAM = "steel:odd_results"


def main() -> None:
    """Load config, build pipeline, run."""
    config = load_config(config_name="config", config_dir=CONFIG_DIR)
    logger.info("Starting {}", config.app_name)

    redis_client: Redis[bytes] = Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=False,
    )

    app = build_application(
        config,
        event_model=SteelInspectionInput,
        postprocessor=YoloPostProcessorBridge(
            conf_threshold=config.pipeline.postprocessor.conf_threshold,
            iou_threshold=config.pipeline.postprocessor.iou_threshold,
            end2end=config.pipeline.postprocessor.end2end,
        ),
        extra_nodes=[
            ("branch_router", CameraBranchRouter()),
            ("result_router", ModelResultRouter()),
            ("even_sink", EvenModelSink(redis_client, EVEN_STREAM)),
            ("odd_sink", OddModelSink(redis_client, ODD_STREAM)),
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
