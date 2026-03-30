"""E2E test harness: push images to Redis, wait for connector to process, collect results."""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from triton_model_tools.detection import (
    Detection,
    FolderInferenceResult,
    ImageDetectionResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    import redis

MILLISECONDS_PER_SECOND = 1000.0
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})
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
    target_size: int = Field(default=640, gt=0)
    end2end: bool = False
    timeout_s: float = Field(default=30.0, gt=0.0)


def _collect_image_paths(image_dir: Path) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        msg = f"No image files found in {image_dir}"
        raise FileNotFoundError(msg)
    logger.info("Found {} images in {}", len(paths), image_dir)
    return paths


def _ensure_consumer_group(
    redis_client: redis.Redis[bytes],
    stream: str,
    group: str,
) -> None:
    import redis as redis_lib  # noqa: PLC0415

    try:
        redis_client.xgroup_create(  # pyright: ignore[reportUnknownMemberType]
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
        import redis as redis_lib  # noqa: PLC0415

        self._config = config
        self._redis: redis.Redis[bytes] = redis_lib.Redis.from_url(
            config.redis_url,
            decode_responses=False,
        )
        _ensure_consumer_group(self._redis, config.output_stream, config.consumer_group)

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
            raw_id: Any = self._redis.xadd(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                name=self._config.input_stream,
                fields={
                    b"data": image_bytes,
                    b"request_id": request_id.encode(),
                    b"model_name": self._config.model_name.encode(),
                    b"image_name": image_path.name.encode(),
                },
            )
            decoded_id: str = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)  # pyright: ignore[reportUnknownArgumentType]
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
            raw_messages: Any = self._redis.xreadgroup(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                groupname=self._config.consumer_group,
                consumername=CONSUMER_NAME,
                streams={self._config.output_stream: ">"},
                count=expected_count - len(results),
                block=block_ms,
            )

            if not raw_messages:
                continue

            for _stream_name, entries in raw_messages:  # pyright: ignore[reportUnknownVariableType]
                for msg_id, fields in entries:  # pyright: ignore[reportUnknownVariableType]
                    result = self._parse_result_message(fields, msg_id)  # pyright: ignore[reportUnknownArgumentType]
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
        image_paths = _collect_image_paths(image_dir)

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
