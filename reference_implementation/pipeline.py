"""Multi-source merger pipeline: 4 sensor streams -> 1 merged output.

Reads four different event types from four Redis streams, normalizes each
into a common MergedReading, and writes to a single output stream.
Demonstrates typed-flow's type-based routing with multiple sources.

Usage:
    PYTHONPATH=. uv run python reference_implementation/pipeline.py
    PYTHONPATH=. uv run python reference_implementation/pipeline.py --config reference_implementation/config.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from loguru import logger
from typed_flow.config.schema import AppConfig
from typed_flow.connectors import RedisThreadSink, RedisThreadSource
from typed_flow.engine.app import Application
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import BaseThreadProcessor, BaseThreadSink

from reference_implementation.models import (
    HumidityReading,
    MergedReading,
    PressureReading,
    TemperatureReading,
    WindReading,
)

if TYPE_CHECKING:
    import types


class TemperatureSource(RedisThreadSource[TemperatureReading]):
    """Reads TemperatureReading from its Redis stream."""


class HumiditySource(RedisThreadSource[HumidityReading]):
    """Reads HumidityReading from its Redis stream."""


class PressureSource(RedisThreadSource[PressureReading]):
    """Reads PressureReading from its Redis stream."""


class WindSource(RedisThreadSource[WindReading]):
    """Reads WindReading from its Redis stream."""


class MergedSink(RedisThreadSink[MergedReading]):
    """Writes MergedReading events to the output Redis stream."""


SensorReading = TemperatureReading | HumidityReading | PressureReading | WindReading


class MergeProcessor(
    BaseThreadProcessor[SensorReading, MergedReading],
):
    """Normalizes any sensor reading into a unified MergedReading."""

    def __call__(self, event: SensorReading) -> MergedReading:  # noqa: D102
        match event:
            case TemperatureReading():
                return MergedReading(
                    flow_meta=event.flow_meta,
                    sensor_id=event.sensor_id,
                    sensor_type="temperature",
                    value=event.celsius,
                    label=f"{event.location} ({event.celsius:.1f}°C)",
                )
            case HumidityReading():
                return MergedReading(
                    flow_meta=event.flow_meta,
                    sensor_id=event.sensor_id,
                    sensor_type="humidity",
                    value=event.humidity_pct,
                    label=f"{event.zone} ({event.humidity_pct:.1f}%)",
                )
            case PressureReading():
                return MergedReading(
                    flow_meta=event.flow_meta,
                    sensor_id=event.sensor_id,
                    sensor_type="pressure",
                    value=event.hpa,
                    label=f"{event.altitude_m:.0f}m ({event.hpa:.1f} hPa)",
                )
            case WindReading():
                return MergedReading(
                    flow_meta=event.flow_meta,
                    sensor_id=event.sensor_id,
                    sensor_type="wind",
                    value=event.speed_mps,
                    label=f"{event.direction_deg}° @ {event.speed_mps:.1f} m/s (gust {event.gust_mps:.1f})",
                )


class ErrorSink(BaseThreadSink[ConnectorError]):
    """Logs deserialization errors from any source connector."""

    def __call__(self, event: ConnectorError) -> None:  # noqa: D102
        logger.error("Connector error on {}: {}", event.stream_name, event.error_message)


SOURCE_CLASSES: list[type[RedisThreadSource]] = [  # type: ignore[type-arg]
    TemperatureSource,
    HumiditySource,
    PressureSource,
    WindSource,
]


def load_config(path: str) -> AppConfig:
    """Load AppConfig from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the merger pipeline."""
    parser = argparse.ArgumentParser(description="Multi-source merger pipeline")
    parser.add_argument(
        "--config",
        default="reference_implementation/config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Build and run the multi-source merger pipeline from YAML config."""
    args = parse_args()
    config = load_config(args.config)
    redis_cfg = config.redis

    expected_sources = 4
    if len(config.sources) != expected_sources:
        logger.error("Expected {} source configs, got {}", expected_sources, len(config.sources))
        sys.exit(1)

    sources = [
        cls(redis=redis_cfg, source=src_cfg, name=f"source_{src_cfg.stream_name}")
        for cls, src_cfg in zip(SOURCE_CLASSES, config.sources, strict=True)
    ]

    sink_cfg = config.sinks[0]
    sink = MergedSink(redis=redis_cfg, sink=sink_cfg, name="merged_sink")

    processors: list[object] = [
        *sources,
        MergeProcessor(),
        sink,
        ErrorSink(),
    ]

    app = Application.build(config, processors=processors)

    def handle_shutdown(signum: int, _frame: types.FrameType | None) -> None:
        logger.info("Received {}, shutting down...", signal.Signals(signum).name)
        app.shutdown()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    stream_names = [s.stream_name for s in config.sources]
    logger.info("Merger pipeline: {} -> {}", stream_names, sink_cfg.stream_name)

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
