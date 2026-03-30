"""Tests for dishka container creation and APP scope resolution."""

from __future__ import annotations

from dishka import make_container
from triton_redis_connector.config.schema import (
    AppConfig,
    InferenceConfig,
    MetricsConfig,
    PreprocessingConfig,
    QueueConfig,
    RedisConnectionConfig,
)
from triton_redis_connector.providers import (
    ConfigProvider,
    ConnectorProvider,
    ConnectorScope,
    EngineProvider,
    InferenceProvider,
    MetricsProvider,
    PreprocessingProvider,
)


def _make_test_container(config: AppConfig | None = None):  # noqa: ANN202
    cfg = config or AppConfig()
    return make_container(
        ConfigProvider(cfg),
        EngineProvider(),
        ConnectorProvider(),
        InferenceProvider(),
        PreprocessingProvider(),
        MetricsProvider(),
        scopes=ConnectorScope,
    )


def test_container_creation() -> None:
    container = _make_test_container()
    container.close()


def test_resolve_app_config() -> None:
    config = AppConfig(app_name="test-app")
    container = _make_test_container(config)
    resolved = container.get(AppConfig)
    assert resolved.app_name == "test-app"
    container.close()


def test_resolve_redis_config() -> None:
    config = AppConfig(redis=RedisConnectionConfig(host="redis-test", port=6380))
    container = _make_test_container(config)
    resolved = container.get(RedisConnectionConfig)
    assert resolved.host == "redis-test"
    assert resolved.port == 6380
    container.close()


def test_resolve_inference_config() -> None:
    config = AppConfig(inference=InferenceConfig(num_inference_workers=4))
    container = _make_test_container(config)
    resolved = container.get(InferenceConfig)
    assert resolved.num_inference_workers == 4
    container.close()


def test_resolve_preprocessing_config() -> None:
    container = _make_test_container()
    resolved = container.get(PreprocessingConfig)
    assert resolved.default_target_size == (640, 640)
    container.close()


def test_resolve_queue_config() -> None:
    config = AppConfig(queues=QueueConfig(maxsize=500))
    container = _make_test_container(config)
    resolved = container.get(QueueConfig)
    assert resolved.maxsize == 500
    container.close()


def test_resolve_metrics_config() -> None:
    container = _make_test_container()
    resolved = container.get(MetricsConfig)
    assert resolved.output_stream == "metrics"
    container.close()


def test_config_provider_returns_same_instance() -> None:
    config = AppConfig()
    container = _make_test_container(config)
    first = container.get(AppConfig)
    second = container.get(AppConfig)
    assert first is second
    container.close()


def test_sub_configs_derive_from_app_config() -> None:
    config = AppConfig(
        redis=RedisConnectionConfig(host="custom-host"),
        inference=InferenceConfig(grpc_url="triton:8001"),
    )
    container = _make_test_container(config)
    app = container.get(AppConfig)
    redis = container.get(RedisConnectionConfig)
    inference = container.get(InferenceConfig)
    assert redis.host == app.redis.host
    assert inference.grpc_url == app.inference.grpc_url
    container.close()
