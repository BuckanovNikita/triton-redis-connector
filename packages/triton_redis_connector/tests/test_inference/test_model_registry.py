"""Tests for ModelVersionTracker with mock Triton client."""

from __future__ import annotations

import time
from dataclasses import dataclass

from triton_redis_connector.inference.model_registry import ModelVersionTracker


@dataclass
class FakeModelInfo:
    name: str
    version: str
    state: str


class FakeTritonClient:
    def __init__(self, models: list[FakeModelInfo] | None = None) -> None:
        self.models: list[FakeModelInfo] = models or []
        self.poll_count = 0
        self.should_fail = False

    def get_model_repository_index(self) -> list[FakeModelInfo]:
        self.poll_count += 1
        if self.should_fail:
            msg = "connection refused"
            raise ConnectionError(msg)
        return self.models


class TestModelVersionTracker:
    def test_get_latest_version_from_single_model(self) -> None:
        client = FakeTritonClient(
            models=[
                FakeModelInfo(name="yolo11s", version="1", state="READY"),
                FakeModelInfo(name="yolo11s", version="2", state="READY"),
            ],
        )
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        assert tracker.get_latest_version("yolo11s") == "2"

    def test_get_latest_version_returns_none_for_unknown_model(self) -> None:
        client = FakeTritonClient(models=[])
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        assert tracker.get_latest_version("unknown") is None

    def test_is_model_ready(self) -> None:
        client = FakeTritonClient(
            models=[
                FakeModelInfo(name="yolo11s", version="1", state="READY"),
                FakeModelInfo(name="yolo11m", version="1", state="LOADING"),
            ],
        )
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        assert tracker.is_model_ready("yolo11s")
        assert not tracker.is_model_ready("yolo11m")
        assert not tracker.is_model_ready("nonexistent")

    def test_version_sorting_numeric(self) -> None:
        client = FakeTritonClient(
            models=[
                FakeModelInfo(name="model", version="10", state="READY"),
                FakeModelInfo(name="model", version="2", state="READY"),
                FakeModelInfo(name="model", version="1", state="READY"),
            ],
        )
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        assert tracker.get_latest_version("model") == "10"

    def test_multiple_models_tracked(self) -> None:
        client = FakeTritonClient(
            models=[
                FakeModelInfo(name="yolo11s", version="1", state="READY"),
                FakeModelInfo(name="yolo11m", version="3", state="READY"),
                FakeModelInfo(name="yolo11m", version="2", state="READY"),
            ],
        )
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        assert tracker.get_latest_version("yolo11s") == "1"
        assert tracker.get_latest_version("yolo11m") == "3"

    def test_polling_updates_models(self) -> None:
        client = FakeTritonClient(
            models=[
                FakeModelInfo(name="yolo11s", version="1", state="READY"),
            ],
        )
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        assert tracker.get_latest_version("yolo11s") == "1"

        client.models = [
            FakeModelInfo(name="yolo11s", version="1", state="READY"),
            FakeModelInfo(name="yolo11s", version="2", state="READY"),
        ]
        tracker._poll_once()

        assert tracker.get_latest_version("yolo11s") == "2"

    def test_poll_failure_preserves_existing_data(self) -> None:
        client = FakeTritonClient(
            models=[
                FakeModelInfo(name="yolo11s", version="1", state="READY"),
            ],
        )
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        client.should_fail = True
        tracker._poll_once()

        assert tracker.get_latest_version("yolo11s") == "1"
        assert tracker.is_model_ready("yolo11s")

    def test_background_thread_polls(self) -> None:
        client = FakeTritonClient(
            models=[
                FakeModelInfo(name="yolo11s", version="1", state="READY"),
            ],
        )
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=0.05)
        tracker.start()

        time.sleep(0.2)
        tracker.stop()

        assert client.poll_count >= 2
        assert tracker.is_model_ready("yolo11s")

    def test_start_stop_idempotent(self) -> None:
        client = FakeTritonClient(models=[])
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=0.05)

        tracker.start()
        tracker.start()
        tracker.stop()
        tracker.stop()

    def test_dict_format_model_info(self) -> None:
        client = FakeTritonClient(models=[])
        client.get_model_repository_index = lambda: [  # type: ignore[assignment]
            {"name": "yolo11s", "version": "1", "state": "READY"},
            {"name": "yolo11s", "version": "2", "state": "READY"},
        ]
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        assert tracker.get_latest_version("yolo11s") == "2"
        assert tracker.is_model_ready("yolo11s")

    def test_empty_version_string_handled(self) -> None:
        client = FakeTritonClient(
            models=[
                FakeModelInfo(name="yolo11s", version="", state="READY"),
            ],
        )
        tracker = ModelVersionTracker(triton_client=client, poll_interval_s=10.0)
        tracker._poll_once()

        assert tracker.is_model_ready("yolo11s")
        assert tracker.get_latest_version("yolo11s") is None
