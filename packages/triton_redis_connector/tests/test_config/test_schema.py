"""Tests for pydantic config models: creation, defaults, validation, serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from triton_redis_connector.config.schema import (
    AppConfig,
    DecoderType,
    InferenceConfig,
    MetricsConfig,
    NodeConfig,
    NodeVariant,
    PreprocessingConfig,
    QueueConfig,
    RedisConnectionConfig,
    SinkConfig,
    SourceConfig,
)


class TestRedisConnectionConfig:
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


class TestSourceConfig:
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
    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="png"):
            DecoderType("png")


class TestInferenceConfig:
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
    def test_zero_workers_rejected(self) -> None:
        with pytest.raises(ValidationError, match="decoder_pool_workers"):
            PreprocessingConfig(decoder_pool_workers=0)

    def test_serialization_roundtrip(self) -> None:
        cfg = PreprocessingConfig(decoder=DecoderType.TURBOJPEG, default_target_size=(416, 416))
        data = cfg.model_dump()
        restored = PreprocessingConfig.model_validate(data)
        assert restored == cfg


class TestQueueConfig:
    def test_zero_maxsize_rejected(self) -> None:
        with pytest.raises(ValidationError, match="maxsize"):
            QueueConfig(maxsize=0)


class TestMetricsConfig:
    def test_zero_poll_interval_rejected(self) -> None:
        with pytest.raises(ValidationError, match="triton_poll_interval_s"):
            MetricsConfig(triton_poll_interval_s=0.0)

    def test_serialization_roundtrip(self) -> None:
        cfg = MetricsConfig(output_stream="custom")
        data = cfg.model_dump()
        restored = MetricsConfig.model_validate(data)
        assert restored == cfg


class TestNodeConfig:
    def test_pool_size_rejected_for_non_pool_variant(self) -> None:
        with pytest.raises(ValidationError, match="pool_size"):
            NodeConfig(
                name="proc",
                class_path="pkg.Proc",
                variant=NodeVariant.THREAD_PROCESSOR,
                pool_size=4,
            )

    def test_pool_size_allowed_for_pool_variant(self) -> None:
        cfg = NodeConfig(
            name="proc",
            class_path="pkg.Proc",
            variant=NodeVariant.THREAD_POOL_PROCESSOR,
            pool_size=4,
        )
        assert cfg.pool_size == 4

    def test_serialization_roundtrip(self) -> None:
        cfg = NodeConfig(
            name="extractor",
            class_path="pkg.Extractor",
            variant=NodeVariant.THREAD_PROCESSOR,
            params={"model_class": "pkg.Model"},
            tracked=True,
        )
        data = cfg.model_dump()
        restored = NodeConfig.model_validate(data)
        assert restored == cfg


class TestAppConfig:
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
            nodes=[
                NodeConfig(
                    name="extractor",
                    class_path="pkg.Extractor",
                    variant=NodeVariant.THREAD_PROCESSOR,
                ),
            ],
            inference=InferenceConfig(grpc_url="triton:8001"),
            preprocessing=PreprocessingConfig(decoder=DecoderType.TURBOJPEG),
            queues=QueueConfig(maxsize=500),
            metrics=MetricsConfig(output_stream="app_metrics"),
        )
        data = cfg.model_dump()
        restored = AppConfig.model_validate(data)
        assert restored == cfg
