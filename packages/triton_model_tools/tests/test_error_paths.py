"""Tests for error paths: connection failures, invalid input, service errors, timeouts."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from triton_model_tools.testing.e2e_test import ConnectorE2ETestHarness, E2ETestConfig
from triton_shared_models.preprocessing import LetterBoxPreprocessor as YoloPreprocessor


class TestRedisConnectionFailure:
    def test_from_url_raises_connection_error(self) -> None:
        with patch("redis.Redis.from_url", side_effect=ConnectionError("Connection refused")):
            config = E2ETestConfig(redis_url="redis://nonexistent:6379")
            with pytest.raises(ConnectionError, match="Connection refused"):
                ConnectorE2ETestHarness(config)

    def test_xgroup_create_raises_non_busygroup(self) -> None:
        import redis

        mock_client = MagicMock()
        mock_client.xgroup_create.side_effect = redis.exceptions.ResponseError("ERR unknown command")

        with patch("redis.Redis.from_url", return_value=mock_client):
            config = E2ETestConfig()
            with pytest.raises(redis.exceptions.ResponseError, match="ERR unknown command"):
                ConnectorE2ETestHarness(config)


class TestInvalidImagePreprocessing:
    def test_zero_dimension_image(self) -> None:
        preprocessor = YoloPreprocessor(target_size=640)
        zero_image = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(Exception):  # noqa: B017, PT011
            preprocessor.preprocess(zero_image)

    def test_single_channel_image(self) -> None:
        preprocessor = YoloPreprocessor(target_size=640)
        mono_image = np.zeros((100, 100, 1), dtype=np.uint8)
        with pytest.raises(Exception):  # noqa: B017, PT011
            preprocessor.preprocess(mono_image)


class TestTritonInferenceFailure:
    @patch("triton_model_tools.inference.client.grpcclient")
    def test_infer_raises_exception(self, mock_grpc: MagicMock) -> None:
        from triton_model_tools.inference.client import TritonInferenceClient

        mock_client_instance = MagicMock()
        mock_grpc.InferenceServerClient.return_value = mock_client_instance
        mock_grpc.InferInput.return_value = MagicMock()
        mock_grpc.InferRequestedOutput.return_value = MagicMock()

        mock_client_instance.infer.side_effect = RuntimeError("Triton inference failed: model not found")

        client = TritonInferenceClient(url="localhost:8001")
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)

        with pytest.raises(RuntimeError, match="model not found"):
            client.infer_raw("nonexistent_model", tensor, ["output0"])


class TestE2ETimeoutHandling:
    @patch("redis.Redis.from_url")
    def test_collect_results_timeout_returns_empty(self, mock_from_url: MagicMock) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client
        mock_client.xreadgroup.return_value = []

        config = E2ETestConfig(timeout_s=0.5)
        harness = ConnectorE2ETestHarness(config)

        results = harness.collect_results(expected_count=5, timeout_s=0.5)

        assert len(results) == 0

    @patch("redis.Redis.from_url")
    def test_collect_results_partial(self, mock_from_url: MagicMock) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        single_result = [
            (
                b"detection_results",
                [
                    (
                        b"msg-1",
                        {
                            b"data": json.dumps(
                                {
                                    "message_id": "img.jpg",
                                    "model_name": "yolo11s",
                                    "inference_time_ms": 10.0,
                                    "detections": [],
                                }
                            ).encode(),
                        },
                    ),
                ],
            ),
        ]
        call_count = 0

        def xreadgroup_side_effect(**_kwargs: object) -> list[object]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return single_result
            return []

        mock_client.xreadgroup.side_effect = xreadgroup_side_effect

        config = E2ETestConfig(timeout_s=1.0)
        harness = ConnectorE2ETestHarness(config)

        results = harness.collect_results(expected_count=3, timeout_s=1.0)

        assert len(results) == 1
        assert results[0].image_path == "img.jpg"
