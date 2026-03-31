"""Tests for MetricEvent model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from triton_redis_connector.metrics import MetricEvent


class TestMetricEvent:
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
