"""E2E test harness: push images to Redis, wait for connector to process, collect results."""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

import redis as redis_lib
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from triton_model_tools.inference.detection_io import collect_image_paths
from triton_model_tools.models.detection import (
    Detection,
    FolderInferenceResult,
    ImageDetectionResult,
)

MILLISECONDS_PER_SECOND = 1000.0

if TYPE_CHECKING:
    from pathlib import Path

    import redis

POLL_INTERVAL_S = 0.5
CONSUMER_NAME = "e2e-harness"


class E2ETestConfig(BaseModel):
    """Configuration for the E2E test harness."""

    model_config = ConfigDict(frozen=True)

    redis_url: str = "redis://redis:6379"
    input_stream: str = "camera_images"
    output_stream: str = "detection_results"
    model_name: str = "yolo11s"
    consumer_group: str = "e2e-test"
    target_size: int | tuple[int, int] = Field(default=640)
    end2end: bool = False
    timeout_s: float = Field(default=30.0, gt=0.0)

    @field_validator("target_size")
    @classmethod
    def _validate_target_size(cls, v: int | tuple[int, int]) -> int | tuple[int, int]:
        dims = (v, v) if isinstance(v, int) else v
        for dim in dims:
            if dim <= 0:
                msg = f"Image dimension {dim} must be positive"
                raise ValueError(msg)
        return v


def ensure_consumer_group(
    redis_client: redis.Redis[bytes],
    stream: str,
    group: str,
) -> None:
    """Create a Redis consumer group, ignoring BUSYGROUP if it already exists."""
    try:
        redis_client.xgroup_create(
            name=stream,
            groupname=group,
            id="0",
            mkstream=True,
        )
        logger.info("Created consumer group {} on stream {}", group, stream)
    except redis_lib.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
        logger.debug("Consumer group {} already exists on stream {}", group, stream)


def _deserialize_detections(raw_detections: str) -> list[Detection]:
    parsed: list[dict[str, object]] = json.loads(raw_detections)
    return [Detection.model_validate(d) for d in parsed]


class ConnectorE2ETestHarness:
    """Drives E2E tests: push images -> Redis -> connector -> Triton -> Redis -> collect results."""

    def __init__(self, config: E2ETestConfig) -> None:
        """Initialize harness with config, connect to Redis, and create consumer group."""
        self._config = config
        self._redis: redis.Redis[bytes] = redis_lib.Redis.from_url(
            config.redis_url,
            decode_responses=False,
        )
        ensure_consumer_group(self._redis, config.output_stream, config.consumer_group)

    @property
    def config(self) -> E2ETestConfig:
        """Return the harness configuration."""
        return self._config

    def push_test_images(self, image_paths: list[Path]) -> list[str]:
        """Push images to Redis input stream via XADD. Returns list of message IDs."""
        message_ids: list[str] = []
        for image_path in image_paths:
            image_bytes = image_path.read_bytes()
            request_id = uuid.uuid4().hex
            raw_id: Any = self._redis.xadd(
                name=self._config.input_stream,
                fields={
                    b"data": image_bytes,
                    b"request_id": request_id.encode(),
                    b"model_name": self._config.model_name.encode(),
                    b"image_name": image_path.name.encode(),
                },
            )
            decoded_id: str = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            message_ids.append(decoded_id)
            logger.debug(
                "Pushed image {} as message {} (request_id={})",
                image_path.name,
                decoded_id,
                request_id,
            )

        logger.info(
            "Pushed {} images to stream {}",
            len(message_ids),
            self._config.input_stream,
        )
        return message_ids

    def collect_results(
        self,
        expected_count: int,
        timeout_s: float | None = None,
    ) -> list[ImageDetectionResult]:
        """Read results from Redis output stream. Blocks until expected_count results or timeout."""
        effective_timeout = timeout_s if timeout_s is not None else self._config.timeout_s
        results: list[ImageDetectionResult] = []
        deadline = time.monotonic() + effective_timeout
        block_ms = int(POLL_INTERVAL_S * MILLISECONDS_PER_SECOND)

        while len(results) < expected_count and time.monotonic() < deadline:
            raw_messages: Any = self._redis.xreadgroup(
                groupname=self._config.consumer_group,
                consumername=CONSUMER_NAME,
                streams={self._config.output_stream: ">"},
                count=expected_count - len(results),
                block=block_ms,
            )

            if not raw_messages:
                continue

            for _stream_name, entries in raw_messages:
                for msg_id, fields in entries:
                    result = self._parse_result_message(fields, msg_id)
                    results.append(result)
                    self._redis.xack(  # type: ignore[no-untyped-call]
                        self._config.output_stream,
                        self._config.consumer_group,
                        msg_id,
                    )

        if len(results) < expected_count:
            logger.warning(
                "Timeout: collected {}/{} results in {:.1f}s",
                len(results),
                expected_count,
                effective_timeout,
            )
        else:
            logger.info("Collected all {} results", len(results))

        return results

    def _parse_result_message(
        self,
        fields: dict[bytes, bytes],
        msg_id: bytes,
    ) -> ImageDetectionResult:
        data_field = fields.get(b"data")
        if data_field is not None:
            return self._parse_pipeline_format(data_field, fields, msg_id)
        return self._parse_legacy_format(fields, msg_id)

    def _parse_pipeline_format(
        self,
        data_field: bytes,
        fields: dict[bytes, bytes],
        msg_id: bytes,
    ) -> ImageDetectionResult:
        """Parse pipeline output where DetectionResult JSON is in the 'data' field."""
        parsed: dict[str, Any] = json.loads(data_field)
        detections_raw = parsed.get("detections_json", "[]")
        detections = _deserialize_detections(detections_raw)
        model_name = parsed.get("model_name", self._config.model_name)
        inference_time_ms = float(parsed.get("inference_time_ms", 0.0))
        source_stream = fields.get(b"source_stream", b"").decode()
        image_name = parsed.get("message_id", "") or source_stream or msg_id.decode()

        logger.debug(
            "Parsed pipeline result {}: {} detections",
            msg_id.decode(),
            len(detections),
        )

        return ImageDetectionResult(
            image_path=image_name,
            model_name=model_name,
            detections=detections,
            inference_time_ms=inference_time_ms,
        )

    def _parse_legacy_format(
        self,
        fields: dict[bytes, bytes],
        msg_id: bytes,
    ) -> ImageDetectionResult:
        """Parse legacy format with individual fields (image_name, detections, etc.)."""
        image_name = fields.get(b"image_name", b"unknown").decode()
        model_name = fields.get(b"model_name", self._config.model_name.encode()).decode()
        inference_time_str = fields.get(b"inference_time_ms", b"0.0").decode()
        detections_raw = fields.get(b"detections", b"[]").decode()

        detections = _deserialize_detections(detections_raw)
        inference_time_ms = float(inference_time_str)

        logger.debug(
            "Parsed result {}: {} detections for {}",
            msg_id.decode(),
            len(detections),
            image_name,
        )

        return ImageDetectionResult(
            image_path=image_name,
            model_name=model_name,
            detections=detections,
            inference_time_ms=inference_time_ms,
        )

    def run_through_connector(self, image_dir: Path) -> FolderInferenceResult:
        """Full E2E: push images -> wait for connector to process -> collect results."""
        image_paths = collect_image_paths(image_dir)

        total_start = time.perf_counter()
        self.push_test_images(image_paths)
        results = self.collect_results(expected_count=len(image_paths))
        total_time_s = time.perf_counter() - total_start

        avg_time_ms = (total_time_s * MILLISECONDS_PER_SECOND / len(results)) if results else 0.0

        logger.info(
            "E2E test complete: {} images, {:.2f}s total, {:.1f}ms avg",
            len(results),
            total_time_s,
            avg_time_ms,
        )

        return FolderInferenceResult(
            model_name=self._config.model_name,
            results=results,
            total_time_s=total_time_s,
            avg_time_ms=avg_time_ms,
        )
