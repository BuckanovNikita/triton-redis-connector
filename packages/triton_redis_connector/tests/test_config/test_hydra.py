"""Tests for Hydra configuration loading and composition."""

from __future__ import annotations

from pathlib import Path

import pytest
from triton_redis_connector.config.hydra_app import load_config
from triton_redis_connector.config.schema import (
    AppConfig,
    DecoderPoolType,
    DecoderType,
)

CONF_DIR = Path(__file__).resolve().parent.parent.parent / "triton_redis_connector" / "conf"


class TestLoadConfigDefaults:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REDIS_HOST", raising=False)
        monkeypatch.delenv("REDIS_PORT", raising=False)
        monkeypatch.delenv("TRITON_GRPC_URL", raising=False)
        monkeypatch.delenv("APP_NAME", raising=False)

    def test_loads_default_config(self) -> None:
        cfg = load_config(config_dir=CONF_DIR)
        assert isinstance(cfg, AppConfig)

    def test_default_app_name(self) -> None:
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.app_name == "triton-connector"

    def test_default_redis_settings(self) -> None:
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.redis.host == "localhost"
        assert cfg.redis.port == 6379
        assert cfg.redis.db == 0
        assert cfg.redis.decode_responses is False

    def test_default_inference_settings(self) -> None:
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.inference.grpc_url == "localhost:8001"
        assert cfg.inference.num_inference_workers == 2
        assert cfg.inference.max_concurrent_per_worker == 8
        assert cfg.inference.max_retry_attempts == 3
        assert cfg.inference.retry_base_delay_s == 0.1
        assert cfg.inference.retry_backoff_multiplier == 4.0
        assert cfg.inference.model_version_poll_interval_s == 5.0

    def test_default_preprocessing_settings(self) -> None:
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.preprocessing.decoder is DecoderType.OPENCV
        assert cfg.preprocessing.decoder_pool_type is DecoderPoolType.THREAD
        assert cfg.preprocessing.decoder_pool_workers == 4
        assert cfg.preprocessing.default_target_size == (640, 640)

    def test_default_metrics_settings(self) -> None:
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.metrics.output_stream == "metrics"
        assert cfg.metrics.triton_poll_interval_s == 10.0
        assert cfg.metrics.error_flush_interval_s == 30.0
        assert cfg.metrics.transit_time_enabled is True

    def test_default_queue_settings(self) -> None:
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.queues.maxsize == 1000

    def test_empty_sources_sinks_processors(self) -> None:
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.sources == []
        assert cfg.sinks == []
        assert cfg.processors == []


class TestLoadConfigOverrides:
    def test_override_app_name(self) -> None:
        cfg = load_config(config_dir=CONF_DIR, overrides=["app_name=my-app"])
        assert cfg.app_name == "my-app"

    def test_override_redis_host(self) -> None:
        cfg = load_config(config_dir=CONF_DIR, overrides=["redis.host=redis-server"])
        assert cfg.redis.host == "redis-server"

    def test_override_redis_port(self) -> None:
        cfg = load_config(config_dir=CONF_DIR, overrides=["redis.port=6380"])
        assert cfg.redis.port == 6380

    def test_override_inference_grpc_url(self) -> None:
        cfg = load_config(config_dir=CONF_DIR, overrides=["inference.grpc_url=triton:8001"])
        assert cfg.inference.grpc_url == "triton:8001"

    def test_override_inference_workers(self) -> None:
        cfg = load_config(config_dir=CONF_DIR, overrides=["inference.num_inference_workers=8"])
        assert cfg.inference.num_inference_workers == 8

    def test_override_preprocessing_decoder(self) -> None:
        cfg = load_config(config_dir=CONF_DIR, overrides=["preprocessing.decoder=turbojpeg"])
        assert cfg.preprocessing.decoder is DecoderType.TURBOJPEG

    def test_override_queue_maxsize(self) -> None:
        cfg = load_config(config_dir=CONF_DIR, overrides=["queues.maxsize=500"])
        assert cfg.queues.maxsize == 500

    def test_multiple_overrides(self) -> None:
        cfg = load_config(
            config_dir=CONF_DIR,
            overrides=[
                "app_name=test-pipeline",
                "redis.host=redis",
                "inference.grpc_url=triton:8001",
                "preprocessing.decoder=turbojpeg",
            ],
        )
        assert cfg.app_name == "test-pipeline"
        assert cfg.redis.host == "redis"
        assert cfg.inference.grpc_url == "triton:8001"
        assert cfg.preprocessing.decoder is DecoderType.TURBOJPEG


class TestLoadConfigEnvVars:
    def test_env_var_override_app_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_NAME", "env-app")
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.app_name == "env-app"

    def test_env_var_override_redis_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REDIS_HOST", "env-redis")
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.redis.host == "env-redis"

    def test_env_var_override_redis_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REDIS_PORT", "6380")
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.redis.port == 6380

    def test_env_var_override_triton_grpc_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRITON_GRPC_URL", "triton:9001")
        cfg = load_config(config_dir=CONF_DIR)
        assert cfg.inference.grpc_url == "triton:9001"


class TestLoadConfigCustomDir:
    def test_custom_config_dir(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text(
            "app_name: custom-app\n"
            "redis:\n"
            "  host: localhost\n"
            "  port: 6379\n"
            "  db: 0\n"
            "  decode_responses: false\n"
            "sources: []\n"
            "sinks: []\n"
            "processors: []\n"
            "inference:\n"
            "  grpc_url: localhost:8001\n"
            "  num_inference_workers: 2\n"
            "  max_concurrent_per_worker: 8\n"
            "  max_retry_attempts: 3\n"
            "  retry_base_delay_s: 0.1\n"
            "  retry_backoff_multiplier: 4.0\n"
            "  model_version_poll_interval_s: 5.0\n"
            "preprocessing:\n"
            "  decoder: opencv\n"
            "  decoder_pool_type: thread\n"
            "  decoder_pool_workers: 4\n"
            "  default_target_size:\n"
            "    - 640\n"
            "    - 640\n"
            "queues:\n"
            "  maxsize: 1000\n"
            "metrics:\n"
            "  output_stream: metrics\n"
            "  triton_poll_interval_s: 10.0\n"
            "  error_flush_interval_s: 30.0\n"
            "  transit_time_enabled: true\n"
        )
        cfg = load_config(config_dir=tmp_path)
        assert cfg.app_name == "custom-app"
