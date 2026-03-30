"""Tests for perf_analyzer integration."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from triton_model_tools.perf_analyzer import (
    PerfAnalyzerConfig,
    PerfAnalyzerResult,
    _build_perf_analyzer_command,
    _parse_perf_analyzer_csv,
    run_perf_analyzer,
)


class TestPerfAnalyzerConfig:
    def test_defaults(self) -> None:
        config = PerfAnalyzerConfig(model_name="yolo11s")

        assert config.model_name == "yolo11s"
        assert config.protocol == "grpc"
        assert config.url == "localhost:8001"
        assert config.concurrency == 1
        assert config.measurement_interval_ms == 5000
        assert config.num_measurements == 10
        assert config.timeout_seconds == 300

    def test_frozen(self) -> None:
        config = PerfAnalyzerConfig(model_name="yolo11s")
        with pytest.raises(Exception):  # noqa: B017, PT011
            config.model_name = "other"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = PerfAnalyzerConfig(
            model_name="yolo26s",
            protocol="http",
            url="triton:8000",
            concurrency=4,
            measurement_interval_ms=10000,
            num_measurements=20,
        )

        assert config.protocol == "http"
        assert config.concurrency == 4

    def test_invalid_concurrency(self) -> None:
        with pytest.raises(Exception):  # noqa: B017, PT011
            PerfAnalyzerConfig(model_name="yolo11s", concurrency=0)


class TestPerfAnalyzerResult:
    def test_creation(self) -> None:
        result = PerfAnalyzerResult(
            avg_latency_ms=5.0,
            p50_latency_ms=4.5,
            p95_latency_ms=8.0,
            p99_latency_ms=12.0,
            throughput_infer_per_sec=200.0,
            concurrency=1,
        )

        assert result.avg_latency_ms == 5.0
        assert result.throughput_infer_per_sec == 200.0


class TestBuildCommand:
    def test_basic_command(self) -> None:
        config = PerfAnalyzerConfig(model_name="yolo11s")

        cmd = _build_perf_analyzer_command(config)

        assert cmd[0] == "perf_analyzer"
        assert "-m" in cmd
        assert cmd[cmd.index("-m") + 1] == "yolo11s"
        assert "-i" in cmd
        assert cmd[cmd.index("-i") + 1] == "grpc"
        assert "--csv" in cmd

    def test_custom_config(self) -> None:
        config = PerfAnalyzerConfig(
            model_name="yolo26s",
            protocol="http",
            url="triton:8000",
            concurrency=4,
        )

        cmd = _build_perf_analyzer_command(config)

        assert cmd[cmd.index("-m") + 1] == "yolo26s"
        assert cmd[cmd.index("-i") + 1] == "http"
        assert cmd[cmd.index("-u") + 1] == "triton:8000"
        assert cmd[cmd.index("--concurrency-range") + 1] == "4"


SAMPLE_CSV_OUTPUT = """\
Concurrency,Inferences/Second,Client Send,p50 latency,p95 latency,p99 latency
1,195.5,5120,4800,8200,12000
"""


class TestParseCsv:
    def test_parses_valid_csv(self) -> None:
        result = _parse_perf_analyzer_csv(SAMPLE_CSV_OUTPUT)

        assert isinstance(result, PerfAnalyzerResult)
        assert result.throughput_infer_per_sec == pytest.approx(195.5)
        assert result.avg_latency_ms == pytest.approx(5.12)
        assert result.p50_latency_ms == pytest.approx(4.8)
        assert result.p95_latency_ms == pytest.approx(8.2)
        assert result.p99_latency_ms == pytest.approx(12.0)
        assert result.concurrency == 1

    def test_empty_csv_raises(self) -> None:
        with pytest.raises(ValueError, match="no data rows"):
            _parse_perf_analyzer_csv("Concurrency,Inferences/Second\n")

    def test_missing_column_raises(self) -> None:
        bad_csv = "Concurrency,Inferences/Second\n1,100\n"
        with pytest.raises(ValueError, match="Missing expected column"):
            _parse_perf_analyzer_csv(bad_csv)


class TestRunPerfAnalyzer:
    @patch("triton_model_tools.perf_analyzer.subprocess.run")
    def test_successful_run(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["perf_analyzer"],
            returncode=0,
            stdout=SAMPLE_CSV_OUTPUT,
            stderr="",
        )

        config = PerfAnalyzerConfig(model_name="yolo11s")
        result = run_perf_analyzer(config)

        assert isinstance(result, PerfAnalyzerResult)
        assert result.throughput_infer_per_sec == pytest.approx(195.5)
        mock_run.assert_called_once()

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "perf_analyzer"
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["check"] is True

    @patch("triton_model_tools.perf_analyzer.subprocess.run")
    def test_timeout_propagates(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="perf_analyzer", timeout=300)

        config = PerfAnalyzerConfig(model_name="yolo11s")

        with pytest.raises(subprocess.TimeoutExpired):
            run_perf_analyzer(config)

    @patch("triton_model_tools.perf_analyzer.subprocess.run")
    def test_nonzero_exit_raises(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="perf_analyzer",
            stderr="Error: model not found",
        )

        config = PerfAnalyzerConfig(model_name="yolo11s")

        with pytest.raises(subprocess.CalledProcessError):
            run_perf_analyzer(config)
