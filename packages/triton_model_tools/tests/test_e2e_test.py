"""Tests for E2E test harness."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from triton_model_tools.models.detection import (
    FolderInferenceResult,
    ImageDetectionResult,
)
from triton_model_tools.testing.e2e_test import (
    ConnectorE2ETestHarness,
    E2ETestConfig,
    _deserialize_detections,
)


class TestE2ETestConfig:
    def test_invalid_target_size(self) -> None:
        with pytest.raises(Exception):  # noqa: B017, PT011
            E2ETestConfig(target_size=0)

    def test_invalid_timeout(self) -> None:
        with pytest.raises(Exception):  # noqa: B017, PT011
            E2ETestConfig(timeout_s=0.0)


class TestDeserializeDetections:
    def test_valid_detections(self) -> None:
        raw = json.dumps(
            [
                {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0, "confidence": 0.9, "class_id": 0},
                {"x1": 50.0, "y1": 60.0, "x2": 150.0, "y2": 160.0, "confidence": 0.8, "class_id": 1},
            ]
        )

        detections = _deserialize_detections(raw)

        assert len(detections) == 2
        assert detections[0].confidence == pytest.approx(0.9)
        assert detections[1].class_id == 1

    def test_empty_list(self) -> None:
        detections = _deserialize_detections("[]")

        assert detections == []


class TestConnectorE2ETestHarness:
    @patch("redis.Redis.from_url")
    def test_creation(self, mock_from_url: MagicMock) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        config = E2ETestConfig(
            redis_url="redis://localhost:6379",
            model_name="yolo11s",
        )
        harness = ConnectorE2ETestHarness(config)

        assert harness.config is config
        mock_from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=False,
        )
        mock_client.xgroup_create.assert_called_once_with(
            name="detection_results",
            groupname="e2e-test",
            id="0",
            mkstream=True,
        )

    @patch("redis.Redis.from_url")
    def test_creation_group_already_exists(self, mock_from_url: MagicMock) -> None:
        import redis

        mock_client = MagicMock()
        mock_client.xgroup_create.side_effect = redis.exceptions.ResponseError("BUSYGROUP already exists")
        mock_from_url.return_value = mock_client

        config = E2ETestConfig()
        ConnectorE2ETestHarness(config)

    @patch("redis.Redis.from_url")
    def test_push_test_images(self, mock_from_url: MagicMock, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1234567890-0"
        mock_from_url.return_value = mock_client

        config = E2ETestConfig()
        harness = ConnectorE2ETestHarness(config)

        img1 = tmp_path / "test1.jpg"
        img2 = tmp_path / "test2.png"
        img1.write_bytes(b"\xff\xd8\xff\xe0fake-jpg")
        img2.write_bytes(b"\x89PNGfake-png")

        message_ids = harness.push_test_images([img1, img2])

        assert len(message_ids) == 2
        assert all(mid == "1234567890-0" for mid in message_ids)
        assert mock_client.xadd.call_count == 2

        first_call = mock_client.xadd.call_args_list[0]
        assert first_call.kwargs["name"] == "camera_images"
        fields = first_call.kwargs["fields"]
        assert fields["data"] == b"\xff\xd8\xff\xe0fake-jpg"
        assert fields["image_name"] == b"test1.jpg"
        assert fields["model_name"] == b"yolo11s"
        assert "request_id" in fields

    @patch("redis.Redis.from_url")
    def test_collect_results_success(self, mock_from_url: MagicMock) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        detections_json = json.dumps(
            [
                {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0, "confidence": 0.9, "class_id": 0},
            ]
        ).encode()

        mock_client.xreadgroup.return_value = [
            (
                b"detection_results",
                [
                    (
                        b"msg-1",
                        {
                            b"image_name": b"test1.jpg",
                            b"model_name": b"yolo11s",
                            b"inference_time_ms": b"15",
                            b"detections": detections_json,
                        },
                    ),
                ],
            ),
        ]

        config = E2ETestConfig(timeout_s=5.0)
        harness = ConnectorE2ETestHarness(config)

        results = harness.collect_results(expected_count=1)

        assert len(results) == 1
        assert isinstance(results[0], ImageDetectionResult)
        assert results[0].image_path == "test1.jpg"
        assert results[0].model_name == "yolo11s"
        assert results[0].inference_time_ms == 15
        assert len(results[0].detections) == 1
        assert results[0].detections[0].confidence == pytest.approx(0.9)

        mock_client.xack.assert_called_once_with(
            "detection_results",
            "e2e-test",
            b"msg-1",
        )

    @patch("redis.Redis.from_url")
    def test_collect_results_timeout(self, mock_from_url: MagicMock) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client
        mock_client.xreadgroup.return_value = []

        config = E2ETestConfig(timeout_s=0.5)
        harness = ConnectorE2ETestHarness(config)

        results = harness.collect_results(expected_count=3, timeout_s=0.5)

        assert len(results) == 0

    @patch("redis.Redis.from_url")
    def test_collect_results_multiple_batches(self, mock_from_url: MagicMock) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        batch1 = [
            (
                b"detection_results",
                [
                    (
                        b"msg-1",
                        {
                            b"image_name": b"img1.jpg",
                            b"model_name": b"yolo11s",
                            b"inference_time_ms": b"10.0",
                            b"detections": b"[]",
                        },
                    ),
                ],
            ),
        ]
        batch2 = [
            (
                b"detection_results",
                [
                    (
                        b"msg-2",
                        {
                            b"image_name": b"img2.jpg",
                            b"model_name": b"yolo11s",
                            b"inference_time_ms": b"12.0",
                            b"detections": b"[]",
                        },
                    ),
                ],
            ),
        ]

        mock_client.xreadgroup.side_effect = [batch1, batch2]

        config = E2ETestConfig(timeout_s=5.0)
        harness = ConnectorE2ETestHarness(config)

        results = harness.collect_results(expected_count=2)

        assert len(results) == 2
        assert results[0].image_path == "img1.jpg"
        assert results[1].image_path == "img2.jpg"

    @patch("redis.Redis.from_url")
    def test_run_through_connector(self, mock_from_url: MagicMock, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client
        mock_client.xadd.return_value = b"1-0"

        detections_json = json.dumps(
            [
                {"x1": 0.0, "y1": 0.0, "x2": 50.0, "y2": 50.0, "confidence": 0.95, "class_id": 2},
            ]
        ).encode()

        mock_client.xreadgroup.return_value = [
            (
                b"detection_results",
                [
                    (
                        b"msg-1",
                        {
                            b"image_name": b"photo.jpg",
                            b"model_name": b"yolo11s",
                            b"inference_time_ms": b"20.0",
                            b"detections": detections_json,
                        },
                    ),
                ],
            ),
        ]

        (tmp_path / "photo.jpg").write_bytes(b"fake-image")

        config = E2ETestConfig(timeout_s=5.0)
        harness = ConnectorE2ETestHarness(config)
        result = harness.run_through_connector(tmp_path)

        assert isinstance(result, FolderInferenceResult)
        assert result.model_name == "yolo11s"
        assert len(result.results) == 1
        assert result.total_time_s > 0
        assert result.avg_time_ms > 0

    @patch("redis.Redis.from_url")
    def test_run_through_connector_empty_dir(self, mock_from_url: MagicMock, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        config = E2ETestConfig()
        harness = ConnectorE2ETestHarness(config)

        with pytest.raises(FileNotFoundError, match="No image files found"):
            harness.run_through_connector(tmp_path)

    @patch("redis.Redis.from_url")
    def test_parse_result_missing_fields_uses_defaults(self, mock_from_url: MagicMock) -> None:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        config = E2ETestConfig()
        harness = ConnectorE2ETestHarness(config)

        result = harness._parse_result_message(
            fields={},
            msg_id=b"test-msg-id",
        )

        assert result.image_path == "unknown"
        assert result.model_name == "yolo11s"
        assert result.inference_time_ms == pytest.approx(0.0)
        assert result.detections == []


class TestE2ETestCLIParsing:
    def _parse(self, args: list[str]) -> argparse.Namespace:
        from triton_model_tools.cli.e2e_test import build_parser as _build_e2e_test_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        _build_e2e_test_parser(subparsers)
        return parser.parse_args(args)

    def test_requires_image_dir_and_model(self) -> None:
        with pytest.raises(SystemExit, match="2"):
            self._parse(["e2e-test"])

    def test_requires_model(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="2"):
            self._parse(["e2e-test", "--image-dir", str(tmp_path)])

    def test_minimal_args(self, tmp_path: Path) -> None:
        ns = self._parse(["e2e-test", "--image-dir", str(tmp_path), "-m", "yolo11s"])

        assert ns.command == "e2e-test"
        assert ns.image_dir == tmp_path
        assert ns.model == "yolo11s"
        assert ns.redis_url == "redis://redis:6379"
        assert ns.input_stream == "camera_images"
        assert ns.output_stream == "detection_results"
        assert ns.consumer_group == "e2e-test"
        assert ns.target_size == 640
        assert ns.end2end is False
        assert ns.timeout == 30.0

    def test_all_flags(self, tmp_path: Path) -> None:
        ns = self._parse(
            [
                "e2e-test",
                "--image-dir",
                str(tmp_path),
                "-m",
                "yolo26s",
                "--redis-url",
                "redis://localhost:6380",
                "--input-stream",
                "input",
                "--output-stream",
                "output",
                "--consumer-group",
                "grp",
                "--target-size",
                "320",
                "--end2end",
                "--timeout",
                "60.0",
            ]
        )

        assert ns.model == "yolo26s"
        assert ns.redis_url == "redis://localhost:6380"
        assert ns.input_stream == "input"
        assert ns.output_stream == "output"
        assert ns.consumer_group == "grp"
        assert ns.target_size == 320
        assert ns.end2end is True
        assert ns.timeout == 60.0
