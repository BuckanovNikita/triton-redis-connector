"""Tests for MetricEvent model."""

import pytest
from pydantic import ValidationError
from typed_flow.metrics import MetricEvent


class TestMetricEvent:
    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricEvent(  # type: ignore[call-arg]
                component_name="connector-app",
                measure="images_processed",
            )

    def test_serialization_roundtrip(self) -> None:
        event = MetricEvent(
            component_name="connector-app",
            measure="queue_overflow",
            value=1.0,
        )
        data = event.model_dump()
        restored = MetricEvent.model_validate(data)
        assert restored == event
