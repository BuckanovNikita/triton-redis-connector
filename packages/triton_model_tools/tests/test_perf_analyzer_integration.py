"""Integration tests for perf_analyzer with real Triton models.

These tests require:
- Triton server running with models loaded
- perf_analyzer binary available on PATH (typically inside triton-sdk container)

Tests are marked as both @integration and @benchmark so they are skipped
by default and only run when explicitly requested.
"""

from __future__ import annotations

import pytest
from triton_model_tools.benchmarks.perf_analyzer import (
    PerfAnalyzerConfig,
    PerfAnalyzerResult,
    is_perf_analyzer_available,
    run_perf_analyzer,
)

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]

PERF_ANALYZER_SKIP_REASON = (
    "perf_analyzer binary not found on PATH (run inside triton-sdk container or install Triton SDK)"
)


@pytest.fixture
def require_perf_analyzer() -> None:
    if not is_perf_analyzer_available():
        pytest.skip(PERF_ANALYZER_SKIP_REASON)


class TestPerfAnalyzerYolo:
    def test_yolo11s_grpc_concurrency_1(
        self,
        require_perf_analyzer: None,
        triton_client: object,
    ) -> None:
        config = PerfAnalyzerConfig(
            model_name="yolo11s",
            protocol="grpc",
            url="localhost:8001",
            concurrency=1,
            measurement_interval_ms=2000,
            num_measurements=3,
            timeout_seconds=120,
        )

        result = run_perf_analyzer(config)

        assert isinstance(result, PerfAnalyzerResult)
        assert result.throughput_infer_per_sec > 0.0
        assert result.avg_latency_ms > 0.0
        assert result.p50_latency_ms > 0.0
        assert result.p95_latency_ms >= result.p50_latency_ms
        assert result.p99_latency_ms >= result.p95_latency_ms
        assert result.concurrency == 1

    def test_yolo11s_grpc_concurrency_4(
        self,
        require_perf_analyzer: None,
        triton_client: object,
    ) -> None:
        config = PerfAnalyzerConfig(
            model_name="yolo11s",
            protocol="grpc",
            url="localhost:8001",
            concurrency=4,
            measurement_interval_ms=2000,
            num_measurements=3,
            timeout_seconds=120,
        )

        result = run_perf_analyzer(config)

        assert isinstance(result, PerfAnalyzerResult)
        assert result.throughput_infer_per_sec > 0.0
        assert result.concurrency == 4


class TestPerfAnalyzerResNet18:
    def test_resnet18_grpc_concurrency_1(
        self,
        require_perf_analyzer: None,
        triton_client: object,
    ) -> None:
        config = PerfAnalyzerConfig(
            model_name="resnet18",
            protocol="grpc",
            url="localhost:8001",
            concurrency=1,
            measurement_interval_ms=2000,
            num_measurements=3,
            timeout_seconds=120,
        )

        result = run_perf_analyzer(config)

        assert isinstance(result, PerfAnalyzerResult)
        assert result.throughput_infer_per_sec > 0.0
        assert result.avg_latency_ms > 0.0
        assert result.p50_latency_ms > 0.0
        assert result.p95_latency_ms >= result.p50_latency_ms
        assert result.p99_latency_ms >= result.p95_latency_ms
        assert result.concurrency == 1

    def test_resnet18_grpc_concurrency_4(
        self,
        require_perf_analyzer: None,
        triton_client: object,
    ) -> None:
        config = PerfAnalyzerConfig(
            model_name="resnet18",
            protocol="grpc",
            url="localhost:8001",
            concurrency=4,
            measurement_interval_ms=2000,
            num_measurements=3,
            timeout_seconds=120,
        )

        result = run_perf_analyzer(config)

        assert isinstance(result, PerfAnalyzerResult)
        assert result.throughput_infer_per_sec > 0.0
        assert result.concurrency == 4
