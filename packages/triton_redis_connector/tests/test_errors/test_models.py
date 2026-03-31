"""Tests for error event models."""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import ValidationError
from triton_redis_connector.errors import (
    ConnectorError,
    InferenceError,
    PreprocessingError,
)

_VALID_INSTANCES: list[tuple[type, dict[str, Any]]] = [
    (
        ConnectorError,
        {
            "connector_name": "redis-source",
            "stream_name": "stream:input",
            "message_id": None,
            "error_type": "ConnectionError",
            "error_message": "Connection refused",
            "timestamp_ms": 1000,
        },
    ),
    (
        InferenceError,
        {
            "model_name": "yolo11s",
            "model_version": "1",
            "error_type": "InferenceServerException",
            "error_message": "Model not found",
            "retry_count": 3,
            "timestamp_ms": 1000,
        },
    ),
    (
        PreprocessingError,
        {
            "message_id": "1234567890-0",
            "decoder_type": "opencv",
            "error_type": "DecodeError",
            "error_message": "Invalid JPEG header",
            "timestamp_ms": 1000,
        },
    ),
]


@pytest.mark.parametrize(
    ("error_cls", "fields"),
    _VALID_INSTANCES,
    ids=["ConnectorError", "InferenceError", "PreprocessingError"],
)
class TestErrorModels:
    def test_frozen_prevents_mutation(self, error_cls: type, fields: dict[str, Any]) -> None:
        error = error_cls(**fields)
        first_field = next(iter(fields))
        with pytest.raises(ValidationError):
            setattr(error, first_field, "modified")

    def test_missing_required_field_raises(self, error_cls: type, fields: dict[str, Any]) -> None:
        incomplete = {k: v for k, v in fields.items() if k != "error_message"}
        with pytest.raises(ValidationError):
            error_cls(**incomplete)


class TestConnectorError:
    def test_creation_with_all_fields(self) -> None:
        error = ConnectorError(
            connector_name="redis-source",
            stream_name="stream:input",
            message_id="1234567890-0",
            error_type="ConnectionError",
            error_message="Connection refused",
            timestamp_ms=int(time.time() * 1000),
        )
        assert error.connector_name == "redis-source"
        assert error.stream_name == "stream:input"
        assert error.message_id == "1234567890-0"
        assert error.error_type == "ConnectionError"

    def test_message_id_nullable(self) -> None:
        error = ConnectorError(
            connector_name="redis-source",
            stream_name="stream:input",
            message_id=None,
            error_type="TimeoutError",
            error_message="Read timed out",
            timestamp_ms=1000,
        )
        assert error.message_id is None


class TestInferenceError:
    def test_creation_with_all_fields(self) -> None:
        error = InferenceError(
            model_name="yolo11s",
            model_version="1",
            error_type="InferenceServerException",
            error_message="Model not found",
            retry_count=3,
            timestamp_ms=int(time.time() * 1000),
        )
        assert error.model_name == "yolo11s"
        assert error.model_version == "1"
        assert error.retry_count == 3


class TestPreprocessingError:
    def test_creation_with_all_fields(self) -> None:
        error = PreprocessingError(
            message_id="1234567890-0",
            decoder_type="opencv",
            error_type="DecodeError",
            error_message="Invalid JPEG header",
            timestamp_ms=int(time.time() * 1000),
        )
        assert error.message_id == "1234567890-0"
        assert error.decoder_type == "opencv"
