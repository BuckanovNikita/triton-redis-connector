"""Minimal camera inference demo with synthetic data.

Demonstrates the framework's type-based DAG wiring without real
inference — all data is generated in-process.

Usage:
    uv run python examples/camera_inference/run_pipeline.py
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

from loguru import logger
from triton_redis_connector.config.hydra_app import load_config
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry

from examples.camera_inference.processors import CameraSource, FramePreprocessor, ResultSink

CONFIG_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Load config, register synthetic nodes, run."""
    config = load_config(config_name="config", config_dir=CONFIG_DIR)
    logger.info("Starting {}", config.app_name)

    registry = PipelineRegistry()
    registry.register("camera_source", CameraSource())
    registry.register("preprocessor", FramePreprocessor())
    registry.register("result_sink", ResultSink())

    app = Application(config=config, registry=registry, queue_bus=QueueBus(config=config.queues))

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
