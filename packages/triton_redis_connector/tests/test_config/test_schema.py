"""Tests for pydantic config models: creation, defaults, validation, serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from triton_redis_connector.config.schema import (
    AppConfig,
    DecoderPoolType,
    DecoderType,
    InferenceConfig,
    MetricsConfig,
    PreprocessingConfig,
    QueueConfig,
    RedisConnectionConfig,
    SinkConfig,
    SourceConfig,
)


class TestRedisConnectionConfig:
    def test_defaults(self) -> None:
        cfg = RedisConnectionConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.db == 0
        assert cfg.decode_responses is False

    def test_custom_values(self) -> None:
        cfg = RedisConnectionConfig(host="redis", port=6380, db=1, decode_responses=True)
        assert cfg.host == "redis"
        assert cfg.port == 6380
        assert cfg.db == 1
        assert cfg.decode_responses is True

    def test_invalid_port_too_high(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            RedisConnectionConfig(port=70000)

    def test_invalid_port_zero(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            RedisConnectionConfig(port=0)

    def test_negative_db_rejected(self) -> None:
        with pytest.raises(ValidationError, match="db"):
            RedisConnectionConfig(db=-1)

    def test_serialization_roundtrip(self) -> None:
        cfg = RedisConnectionConfig(host="redis-host", port=6380)
        data = cfg.model_dump()
        restored = RedisConnectionConfig.model_validate(data)
        assert restored == cfg

    def test_frozen(self) -> None:
        cfg = RedisConnectionConfig()
        with pytest.raises(ValidationError):
            cfg.host = "other"  # type: ignore[misc]


class TestSourceConfig:
    def test_required_fields(self) -> None:
        cfg = SourceConfig(
            stream_name="camera_images",
            consumer_group="inference_group",
            event_model="myapp.models.CameraImageEvent",
        )
        assert cfg.stream_name == "camera_images"
        assert cfg.consumer_group == "inference_group"
        assert cfg.consumer_name == ""
        assert cfg.event_model == "myapp.models.CameraImageEvent"

    def test_defaults(self) -> None:
        cfg = SourceConfig(
            stream_name="s",
            consumer_group="g",
            event_model="m.M",
        )
        assert cfg.batch_size == 10
        assert cfg.block_ms == 1000
        assert cfg.newest_first_on_recovery is True
        assert cfg.recovery_interleave_ratio == 4
        assert cfg.pending_ttl_s == 60.0

    def test_empty_stream_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="stream_name"):
            SourceConfig(stream_name="", consumer_group="g", event_model="m.M")

    def test_empty_consumer_group_rejected(self) -> None:
        with pytest.raises(ValidationError, match="consumer_group"):
            SourceConfig(stream_name="s", consumer_group="", event_model="m.M")

    def test_empty_event_model_rejected(self) -> None:
        with pytest.raises(ValidationError, match="event_model"):
            SourceConfig(stream_name="s", consumer_group="g", event_model="")

    def test_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="batch_size"):
            SourceConfig(stream_name="s", consumer_group="g", event_model="m.M", batch_size=0)

    def test_block_ms_allows_zero(self) -> None:
        cfg = SourceConfig(stream_name="s", consumer_group="g", event_model="m.M", block_ms=0)
        assert cfg.block_ms == 0

    def test_recovery_interleave_ratio_minimum(self) -> None:
        with pytest.raises(ValidationError, match="recovery_interleave_ratio"):
            SourceConfig(stream_name="s", consumer_group="g", event_model="m.M", recovery_interleave_ratio=0)

    def test_pending_ttl_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="pending_ttl_s"):
            SourceConfig(stream_name="s", consumer_group="g", event_model="m.M", pending_ttl_s=0.0)

    def test_serialization_roundtrip(self) -> None:
        cfg = SourceConfig(
            stream_name="input",
            consumer_group="grp",
            event_model="pkg.Model",
            batch_size=5,
        )
        data = cfg.model_dump()
        restored = SourceConfig.model_validate(data)
        assert restored == cfg


class TestSinkConfig:
    def test_required_fields(self) -> None:
        cfg = SinkConfig(stream_name="detection_results")
        assert cfg.stream_name == "detection_results"
        assert cfg.maxlen == 10_000
        assert cfg.approximate_maxlen is True

    def test_none_maxlen_means_unbounded(self) -> None:
        cfg = SinkConfig(stream_name="out", maxlen=None)
        assert cfg.maxlen is None

    def test_empty_stream_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="stream_name"):
            SinkConfig(stream_name="")

    def test_maxlen_must_be_positive_if_set(self) -> None:
        with pytest.raises(ValidationError, match="maxlen"):
            SinkConfig(stream_name="out", maxlen=0)

    def test_serialization_roundtrip(self) -> None:
        cfg = SinkConfig(stream_name="results", maxlen=5000, approximate_maxlen=False)
        data = cfg.model_dump()
        restored = SinkConfig.model_validate(data)
        assert restored == cfg


class TestDecoderType:
    def test_values(self) -> None:
        assert DecoderType.OPENCV == "opencv"
        assert DecoderType.TURBOJPEG == "turbojpeg"

    def test_from_string(self) -> None:
        assert DecoderType("opencv") is DecoderType.OPENCV
        assert DecoderType("turbojpeg") is DecoderType.TURBOJPEG

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="png"):
            DecoderType("png")


class TestDecoderPoolType:
    def test_values(self) -> None:
        assert DecoderPoolType.THREAD == "thread"
        assert DecoderPoolType.PROCESS == "process"

    def test_from_string(self) -> None:
        assert DecoderPoolType("thread") is DecoderPoolType.THREAD
        assert DecoderPoolType("process") is DecoderPoolType.PROCESS


class TestInferenceConfig:
    def test_defaults(self) -> None:
        cfg = InferenceConfig()
        assert cfg.grpc_url == "localhost:8001"
        assert cfg.num_inference_workers == 2
        assert cfg.max_concurrent_per_worker == 8
        assert cfg.max_retry_attempts == 3
        assert cfg.retry_base_delay_s == 0.1
        assert cfg.retry_backoff_multiplier == 4.0
        assert cfg.model_version_poll_interval_s == 5.0

    def test_custom_values(self) -> None:
        cfg = InferenceConfig(grpc_url="triton:8001", num_inference_workers=4, max_retry_attempts=0)
        assert cfg.grpc_url == "triton:8001"
        assert cfg.num_inference_workers == 4
        assert cfg.max_retry_attempts == 0

    def test_zero_workers_rejected(self) -> None:
        with pytest.raises(ValidationError, match="num_inference_workers"):
            InferenceConfig(num_inference_workers=0)

    def test_negative_retry_base_delay_rejected(self) -> None:
        with pytest.raises(ValidationError, match="retry_base_delay_s"):
            InferenceConfig(retry_base_delay_s=0.0)

    def test_backoff_multiplier_less_than_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="retry_backoff_multiplier"):
            InferenceConfig(retry_backoff_multiplier=0.5)

    def test_serialization_roundtrip(self) -> None:
        cfg = InferenceConfig(grpc_url="triton:8001")
        data = cfg.model_dump()
        restored = InferenceConfig.model_validate(data)
        assert restored == cfg


class TestPreprocessingConfig:
    def test_defaults(self) -> None:
        cfg = PreprocessingConfig()
        assert cfg.decoder is DecoderType.OPENCV
        assert cfg.decoder_pool_type is DecoderPoolType.THREAD
        assert cfg.decoder_pool_workers == 4
        assert cfg.default_target_size == (640, 640)

    def test_turbojpeg_with_process_pool(self) -> None:
        cfg = PreprocessingConfig(
            decoder=DecoderType.TURBOJPEG,
            decoder_pool_type=DecoderPoolType.PROCESS,
            decoder_pool_workers=8,
        )
        assert cfg.decoder is DecoderType.TURBOJPEG
        assert cfg.decoder_pool_type is DecoderPoolType.PROCESS
        assert cfg.decoder_pool_workers == 8

    def test_custom_target_size(self) -> None:
        cfg = PreprocessingConfig(default_target_size=(320, 320))
        assert cfg.default_target_size == (320, 320)

    def test_zero_workers_rejected(self) -> None:
        with pytest.raises(ValidationError, match="decoder_pool_workers"):
            PreprocessingConfig(decoder_pool_workers=0)

    def test_serialization_roundtrip(self) -> None:
        cfg = PreprocessingConfig(decoder=DecoderType.TURBOJPEG, default_target_size=(416, 416))
        data = cfg.model_dump()
        restored = PreprocessingConfig.model_validate(data)
        assert restored == cfg


class TestQueueConfig:
    def test_defaults(self) -> None:
        cfg = QueueConfig()
        assert cfg.maxsize == 1000

    def test_custom_maxsize(self) -> None:
        cfg = QueueConfig(maxsize=500)
        assert cfg.maxsize == 500

    def test_zero_maxsize_rejected(self) -> None:
        with pytest.raises(ValidationError, match="maxsize"):
            QueueConfig(maxsize=0)


class TestMetricsConfig:
    def test_defaults(self) -> None:
        cfg = MetricsConfig()
        assert cfg.output_stream == "metrics"
        assert cfg.triton_poll_interval_s == 10.0
        assert cfg.error_flush_interval_s == 30.0
        assert cfg.transit_time_enabled is True

    def test_custom_values(self) -> None:
        cfg = MetricsConfig(
            output_stream="app_metrics",
            triton_poll_interval_s=5.0,
            error_flush_interval_s=15.0,
            transit_time_enabled=False,
        )
        assert cfg.output_stream == "app_metrics"
        assert cfg.triton_poll_interval_s == 5.0
        assert cfg.error_flush_interval_s == 15.0
        assert cfg.transit_time_enabled is False

    def test_zero_poll_interval_rejected(self) -> None:
        with pytest.raises(ValidationError, match="triton_poll_interval_s"):
            MetricsConfig(triton_poll_interval_s=0.0)

    def test_serialization_roundtrip(self) -> None:
        cfg = MetricsConfig(output_stream="custom")
        data = cfg.model_dump()
        restored = MetricsConfig.model_validate(data)
        assert restored == cfg


class TestAppConfig:
    def test_all_defaults(self) -> None:
        cfg = AppConfig()
        assert cfg.app_name == "triton-connector"
        assert cfg.redis == RedisConnectionConfig()
        assert cfg.sources == []
        assert cfg.sinks == []
        assert cfg.processors == []
        assert cfg.inference == InferenceConfig()
        assert cfg.preprocessing == PreprocessingConfig()
        assert cfg.queues == QueueConfig()
        assert cfg.metrics == MetricsConfig()

    def test_with_sources_and_sinks(self) -> None:
        cfg = AppConfig(
            sources=[
                SourceConfig(
                    stream_name="camera_images",
                    consumer_group="inference_group",
                    event_model="myapp.models.CameraImageEvent",
                ),
            ],
            sinks=[
                SinkConfig(stream_name="detection_results"),
                SinkConfig(stream_name="metrics", maxlen=50000),
            ],
            processors=["myapp.processors.ImageDecodeProcessor"],
        )
        assert len(cfg.sources) == 1
        assert cfg.sources[0].stream_name == "camera_images"
        assert len(cfg.sinks) == 2
        assert cfg.sinks[1].maxlen == 50000
        assert len(cfg.processors) == 1

    def test_full_serialization_roundtrip(self) -> None:
        cfg = AppConfig(
            app_name="test-app",
            redis=RedisConnectionConfig(host="redis", port=6380),
            sources=[
                SourceConfig(
                    stream_name="input",
                    consumer_group="grp",
                    event_model="pkg.Model",
                ),
            ],
            sinks=[SinkConfig(stream_name="output")],
            processors=["pkg.Proc"],
            inference=InferenceConfig(grpc_url="triton:8001"),
            preprocessing=PreprocessingConfig(decoder=DecoderType.TURBOJPEG),
            queues=QueueConfig(maxsize=500),
            metrics=MetricsConfig(output_stream="app_metrics"),
        )
        data = cfg.model_dump()
        restored = AppConfig.model_validate(data)
        assert restored == cfg

    def test_model_dump_produces_plain_dict(self) -> None:
        cfg = AppConfig(
            sources=[
                SourceConfig(
                    stream_name="s",
                    consumer_group="g",
                    event_model="m.M",
                ),
            ],
        )
        data = cfg.model_dump()
        assert isinstance(data, dict)
        assert isinstance(data["sources"], list)
        assert isinstance(data["sources"][0], dict)
        assert data["sources"][0]["stream_name"] == "s"

    def test_nested_config_override(self) -> None:
        cfg = AppConfig(
            redis=RedisConnectionConfig(host="custom-redis"),
            inference=InferenceConfig(num_inference_workers=8),
        )
        assert cfg.redis.host == "custom-redis"
        assert cfg.inference.num_inference_workers == 8
