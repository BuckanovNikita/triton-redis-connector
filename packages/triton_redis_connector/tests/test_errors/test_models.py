"""Tests for error event models."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError
from triton_redis_connector.errors import (
    ConnectorError,
    InferenceError,
    PreprocessingError,
)


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

    def test_frozen_prevents_mutation(self) -> None:
        error = ConnectorError(
            connector_name="redis-source",
            stream_name="stream:input",
            message_id=None,
            error_type="ConnectionError",
            error_message="Connection refused",
            timestamp_ms=1000,
        )
        with pytest.raises(ValidationError):
            error.connector_name = "modified"  # type: ignore[misc]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConnectorError(  # type: ignore[call-arg]
                connector_name="redis-source",
                stream_name="stream:input",
                error_type="ConnectionError",
                timestamp_ms=1000,
            )


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

    def test_frozen_prevents_mutation(self) -> None:
        error = InferenceError(
            model_name="yolo11s",
            model_version="1",
            error_type="InferenceServerException",
            error_message="Model not found",
            retry_count=3,
            timestamp_ms=1000,
        )
        with pytest.raises(ValidationError):
            error.retry_count = 5  # type: ignore[misc]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            InferenceError(  # type: ignore[call-arg]
                model_name="yolo11s",
                error_type="InferenceServerException",
                error_message="Model not found",
                timestamp_ms=1000,
            )


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

    def test_frozen_prevents_mutation(self) -> None:
        error = PreprocessingError(
            message_id="1234567890-0",
            decoder_type="turbojpeg",
            error_type="DecodeError",
            error_message="Truncated file",
            timestamp_ms=1000,
        )
        with pytest.raises(ValidationError):
            error.decoder_type = "opencv"  # type: ignore[misc]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            PreprocessingError(  # type: ignore[call-arg]
                message_id="1234567890-0",
                error_type="DecodeError",
                error_message="Invalid JPEG header",
            )
