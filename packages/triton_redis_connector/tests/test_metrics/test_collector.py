"""Tests for MetricEvent model."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError
from triton_redis_connector.metrics import MetricEvent


class TestMetricEvent:
    def test_creation_with_defaults(self) -> None:
        event = MetricEvent(
            component_name="connector-app",
            measure="images_processed",
            value=42.0,
            timestamp_ms=int(time.time() * 1000),
        )
        assert event.component_name == "connector-app"
        assert event.measure == "images_processed"
        assert event.value == 42.0
        assert event.tags == {}

    def test_creation_with_tags(self) -> None:
        event = MetricEvent(
            component_name="connector-app",
            measure="inference_time_ms",
            value=12.5,
            timestamp_ms=1000,
            tags={"model": "yolo11s", "queue": "preprocessing"},
        )
        assert event.tags == {"model": "yolo11s", "queue": "preprocessing"}

    def test_frozen_prevents_mutation(self) -> None:
        event = MetricEvent(
            component_name="connector-app",
            measure="images_processed",
            value=42.0,
            timestamp_ms=1000,
        )
        with pytest.raises(ValidationError):
            event.value = 99.0  # type: ignore[misc]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricEvent(  # type: ignore[call-arg]
                component_name="connector-app",
                measure="images_processed",
            )

    def test_tags_default_factory_isolation(self) -> None:
        event_a = MetricEvent(
            component_name="app",
            measure="test",
            value=1.0,
            timestamp_ms=1000,
        )
        event_b = MetricEvent(
            component_name="app",
            measure="test",
            value=2.0,
            timestamp_ms=2000,
        )
        assert event_a.tags is not event_b.tags

    def test_serialization_roundtrip(self) -> None:
        event = MetricEvent(
            component_name="connector-app",
            measure="queue_overflow",
            value=1.0,
            timestamp_ms=1709000000000,
            tags={"queue": "inference_input"},
        )
        data = event.model_dump()
        restored = MetricEvent.model_validate(data)
        assert restored == event
