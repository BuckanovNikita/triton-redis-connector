"""Multi-source enrichment pipeline: 4 sensor streams -> 1 enriched output.

Reads four different event types from four Redis streams, enriches each
with random fields, and writes to a single output stream via a union sink.
Demonstrates typed-flow's type-based routing with multiple sources and metrics.

Usage:
    PYTHONPATH=. uv run python reference_implementation/pipeline.py
    PYTHONPATH=. uv run python reference_implementation/pipeline.py --config reference_implementation/config.yaml
"""

import argparse
import random
import signal
import sys
import types
from pathlib import Path

import yaml
from loguru import logger
from typed_flow.config.schema import AppConfig
from typed_flow.connectors import RedisThreadSink, RedisThreadSource
from typed_flow.engine.app import Application
from typed_flow.errors.models import PipelineError
from typed_flow.protocols import BaseThreadProcessor, BaseThreadSink
from typed_flow.protocols.metrics import TimeMetric

from reference_implementation.models import (
    EnrichedHumidity,
    EnrichedPressure,
    EnrichedReading,
    EnrichedTemperature,
    EnrichedWind,
    HumidityReading,
    PressureReading,
    SensorReading,
    TemperatureReading,
    WindReading,
)


class TemperatureSource(RedisThreadSource[TemperatureReading]):
    """Reads TemperatureReading from its Redis stream."""


class HumiditySource(RedisThreadSource[HumidityReading]):
    """Reads HumidityReading from its Redis stream."""


class PressureSource(RedisThreadSource[PressureReading]):
    """Reads PressureReading from its Redis stream."""


class WindSource(RedisThreadSource[WindReading]):
    """Reads WindReading from its Redis stream."""


class EnrichedSink(RedisThreadSink[EnrichedReading]):
    """Writes enriched events to the output Redis stream."""


class EnrichProcessor(
    BaseThreadProcessor[SensorReading, EnrichedReading],
):
    """Enriches any sensor reading with random fields."""

    def __call__(self, event: SensorReading) -> EnrichedReading:  # noqa: D102
        now = TimeMetric.now()
        score = random.random()  # noqa: S311

        match event:
            case TemperatureReading():
                return EnrichedTemperature(
                    flow_meta=event.flow_meta,
                    sensor_id=event.sensor_id,
                    celsius=event.celsius,
                    location=event.location,
                    sensor_source="temperature",
                    random_score=score,
                    processed_at=now,
                )
            case HumidityReading():
                return EnrichedHumidity(
                    flow_meta=event.flow_meta,
                    sensor_id=event.sensor_id,
                    humidity_pct=event.humidity_pct,
                    zone=event.zone,
                    sensor_source="humidity",
                    random_score=score,
                    processed_at=now,
                )
            case PressureReading():
                return EnrichedPressure(
                    flow_meta=event.flow_meta,
                    sensor_id=event.sensor_id,
                    hpa=event.hpa,
                    altitude_m=event.altitude_m,
                    sensor_source="pressure",
                    random_score=score,
                    processed_at=now,
                )
            case WindReading():
                return EnrichedWind(
                    flow_meta=event.flow_meta,
                    sensor_id=event.sensor_id,
                    speed_mps=event.speed_mps,
                    direction_deg=event.direction_deg,
                    gust_mps=event.gust_mps,
                    sensor_source="wind",
                    random_score=score,
                    processed_at=now,
                )


class ErrorSink(BaseThreadSink[PipelineError]):
    """Logs all pipeline errors (connector, processing, etc.)."""

    def __call__(self, event: PipelineError) -> None:  # noqa: D102
        logger.error("[{}] {}: {}", event.node_name, event.error_type, event.error_message)


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
    """Parse CLI arguments for the enrichment pipeline."""
    parser = argparse.ArgumentParser(description="Multi-source enrichment pipeline")
    parser.add_argument(
        "--config",
        default="reference_implementation/config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Build and run the multi-source enrichment pipeline from YAML config."""
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
    sink = EnrichedSink(redis=redis_cfg, sink=sink_cfg, name="enriched_sink")

    processors: list[object] = [
        *sources,
        EnrichProcessor(),
        sink,
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

    stream_names = [s.stream_name for s in config.sources]
    logger.info("Enrichment pipeline: {} -> {}", stream_names, sink_cfg.stream_name)

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
