"""Tests for pipeline benchmark configuration and stats."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from triton_model_tools.benchmarks.pipeline_benchmark import PipelineBenchmarkConfig, PipelineBenchmarkResult


class TestPipelineBenchmarkConfig:
    def test_defaults(self) -> None:
        config = PipelineBenchmarkConfig()
        assert config.redis_url == "redis://redis:6379"
        assert config.model_name == "yolo11s"
        assert config.num_iterations == 3
        assert config.warmup_runs == 1
        assert config.timeout_s == 60.0

    def test_custom_values(self) -> None:
        config = PipelineBenchmarkConfig(
            model_name="yolo26s",
            num_iterations=5,
            warmup_runs=2,
        )
        assert config.model_name == "yolo26s"
        assert config.num_iterations == 5

    def test_frozen(self) -> None:
        config = PipelineBenchmarkConfig()
        with pytest.raises(ValidationError):
            config.model_name = "other"  # type: ignore[misc]


class TestPipelineBenchmarkResult:
    def test_creation(self) -> None:
        config = PipelineBenchmarkConfig()
        result = PipelineBenchmarkResult(
            config=config,
            total_time_s=1.0,
            avg_latency_ms=100.0,
            throughput_fps=10.0,
            p50_latency_ms=90.0,
            p95_latency_ms=150.0,
            p99_latency_ms=200.0,
            num_images=10,
            num_iterations=3,
        )
        assert result.avg_latency_ms == 100.0
        assert result.num_images == 10
