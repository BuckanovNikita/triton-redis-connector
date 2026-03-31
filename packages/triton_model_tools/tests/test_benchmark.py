"""Tests for benchmark engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from triton_model_tools.benchmarks.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    build_full_matrix,
    run_benchmark,
    run_benchmark_matrix,
    run_ultralytics_benchmark,
)
from triton_model_tools.benchmarks.stats import compute_statistics

if TYPE_CHECKING:
    from pathlib import Path


class TestBenchmarkConfig:
    def test_defaults(self) -> None:
        config = BenchmarkConfig(model_name="yolo11s")

        assert config.model_name == "yolo11s"
        assert config.backend == "onnx"
        assert config.preprocess_mode == "raw"
        assert config.triton_url == "localhost:8001"
        assert config.target_size == 640
        assert config.num_images == 100
        assert config.warmup_runs == 5

    def test_frozen(self) -> None:
        config = BenchmarkConfig(model_name="yolo11s")
        with pytest.raises(Exception):  # noqa: B017, PT011
            config.model_name = "other"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = BenchmarkConfig(
            model_name="yolo26s",
            backend="tensorrt",
            preprocess_mode="dali",
            triton_url="triton:8001",
            target_size=320,
            num_images=50,
            warmup_runs=10,
            end2end=True,
        )

        assert config.backend == "tensorrt"
        assert config.preprocess_mode == "dali"
        assert config.target_size == 320
        assert config.end2end is True

    def test_invalid_conf_threshold(self) -> None:
        with pytest.raises(Exception):  # noqa: B017, PT011
            BenchmarkConfig(model_name="yolo11s", conf_threshold=1.5)


class TestBenchmarkResult:
    def test_creation(self) -> None:
        config = BenchmarkConfig(model_name="yolo11s")
        result = BenchmarkResult(
            config=config,
            total_time_s=10.0,
            avg_latency_ms=100.0,
            throughput_fps=10.0,
            p50_latency_ms=95.0,
            p95_latency_ms=150.0,
            p99_latency_ms=200.0,
            num_images=100,
        )

        assert result.total_time_s == 10.0
        assert result.avg_latency_ms == 100.0
        assert result.throughput_fps == 10.0
        assert result.num_images == 100

    def test_frozen(self) -> None:
        config = BenchmarkConfig(model_name="yolo11s")
        result = BenchmarkResult(
            config=config,
            total_time_s=10.0,
            avg_latency_ms=100.0,
            throughput_fps=10.0,
            p50_latency_ms=95.0,
            p95_latency_ms=150.0,
            p99_latency_ms=200.0,
            num_images=100,
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            result.total_time_s = 20.0  # type: ignore[misc]


class TestBenchmarkSuite:
    def _make_result(self, model_name: str = "yolo11s") -> BenchmarkResult:
        config = BenchmarkConfig(model_name=model_name)
        return BenchmarkResult(
            config=config,
            total_time_s=10.0,
            avg_latency_ms=100.0,
            throughput_fps=10.0,
            p50_latency_ms=95.0,
            p95_latency_ms=150.0,
            p99_latency_ms=200.0,
            num_images=100,
        )

    def test_to_json(self) -> None:
        suite = BenchmarkSuite(results=[self._make_result()])

        json_str = suite.to_json()
        parsed = json.loads(json_str)

        assert len(parsed["results"]) == 1
        assert "timestamp" in parsed
        assert "python_version" in parsed
        assert "platform_info" in parsed

    def test_to_csv(self, tmp_path: Path) -> None:
        suite = BenchmarkSuite(
            results=[
                self._make_result("yolo11s"),
                self._make_result("yolo26s"),
            ],
        )

        csv_path = tmp_path / "results.csv"
        suite.to_csv(csv_path)

        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_to_csv_empty(self, tmp_path: Path) -> None:
        suite = BenchmarkSuite(results=[])

        csv_path = tmp_path / "results.csv"
        suite.to_csv(csv_path)

        assert not csv_path.exists()

    def test_json_roundtrip(self) -> None:
        suite = BenchmarkSuite(results=[self._make_result()])

        json_str = suite.to_json()
        restored = BenchmarkSuite.model_validate_json(json_str)

        assert len(restored.results) == 1
        assert restored.results[0].config.model_name == "yolo11s"
        assert restored.results[0].avg_latency_ms == pytest.approx(100.0)

    def test_metadata_populated(self) -> None:
        suite = BenchmarkSuite(results=[])

        assert suite.timestamp != ""
        assert suite.python_version != ""
        assert suite.platform_info != ""


class TestComputeStatistics:
    def test_basic_statistics(self) -> None:
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]

        total_s, avg_ms, throughput, p50, _p95, _p99 = compute_statistics(latencies)

        assert total_s == pytest.approx(0.15)
        assert avg_ms == pytest.approx(30.0)
        assert throughput == pytest.approx(5 / 0.15)
        assert p50 == pytest.approx(30.0)

    def test_single_value(self) -> None:
        latencies = [42.0]

        _total_s, avg_ms, _throughput, p50, p95, p99 = compute_statistics(latencies)

        assert avg_ms == pytest.approx(42.0)
        assert p50 == pytest.approx(42.0)
        assert p95 == pytest.approx(42.0)
        assert p99 == pytest.approx(42.0)


class TestRunBenchmark:
    @patch("triton_model_tools.benchmarks.benchmark.load_image")
    @patch("triton_model_tools.benchmarks.benchmark.TritonInferenceClient")
    def test_runs_benchmark_with_warmup(
        self,
        mock_client_cls: MagicMock,
        mock_load_image: MagicMock,
        tmp_path: Path,
    ) -> None:
        for i in range(5):
            (tmp_path / f"img{i}.jpg").write_bytes(b"fake")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = fake_image

        mock_client = MagicMock()
        mock_client.infer_raw.return_value = {
            "output0": np.zeros((1, 84, 8400), dtype=np.float32),
        }
        mock_client_cls.return_value = mock_client

        config = BenchmarkConfig(
            model_name="yolo11s",
            num_images=5,
            warmup_runs=2,
        )

        result = run_benchmark(config, tmp_path)

        assert isinstance(result, BenchmarkResult)
        assert result.num_images == 5
        assert result.total_time_s > 0
        assert result.avg_latency_ms > 0
        assert result.throughput_fps > 0
        assert mock_client.infer_raw.call_count == 5 + 2

    @patch("triton_model_tools.benchmarks.benchmark.load_image")
    @patch("triton_model_tools.benchmarks.benchmark.TritonInferenceClient")
    def test_no_warmup(
        self,
        mock_client_cls: MagicMock,
        mock_load_image: MagicMock,
        tmp_path: Path,
    ) -> None:
        for i in range(3):
            (tmp_path / f"img{i}.jpg").write_bytes(b"fake")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = fake_image

        mock_client = MagicMock()
        mock_client.infer_raw.return_value = {
            "output0": np.zeros((1, 84, 8400), dtype=np.float32),
        }
        mock_client_cls.return_value = mock_client

        config = BenchmarkConfig(
            model_name="yolo11s",
            num_images=3,
            warmup_runs=0,
        )

        result = run_benchmark(config, tmp_path)

        assert result.num_images == 3
        assert mock_client.infer_raw.call_count == 3

    def test_raises_on_empty_directory(self, tmp_path: Path) -> None:
        config = BenchmarkConfig(model_name="yolo11s")

        with pytest.raises(FileNotFoundError, match="No image files found"):
            run_benchmark(config, tmp_path)


class TestRunBenchmarkMatrix:
    @patch("triton_model_tools.benchmarks.benchmark.load_image")
    @patch("triton_model_tools.benchmarks.benchmark.TritonInferenceClient")
    def test_runs_all_configs(
        self,
        mock_client_cls: MagicMock,
        mock_load_image: MagicMock,
        tmp_path: Path,
    ) -> None:
        for i in range(3):
            (tmp_path / f"img{i}.jpg").write_bytes(b"fake")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = fake_image

        mock_client = MagicMock()
        mock_client.infer_raw.return_value = {
            "output0": np.zeros((1, 84, 8400), dtype=np.float32),
        }
        mock_client_cls.return_value = mock_client

        configs = [
            BenchmarkConfig(model_name="yolo11s", num_images=3, warmup_runs=0),
            BenchmarkConfig(model_name="yolo26s", num_images=3, warmup_runs=0),
        ]

        suite = run_benchmark_matrix(configs, tmp_path)

        assert isinstance(suite, BenchmarkSuite)
        assert len(suite.results) == 2
        assert suite.results[0].config.model_name == "yolo11s"
        assert suite.results[1].config.model_name == "yolo26s"


class TestRunUltralyticsBaseline:
    def test_runs_ultralytics(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"img{i}.jpg").write_bytes(b"fake")

        mock_model = MagicMock()
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with patch("triton_model_tools.benchmarks.benchmark.YOLO", mock_yolo_cls):
            result = run_ultralytics_benchmark("yolo11s.pt", tmp_path, num_images=3)

        assert isinstance(result, BenchmarkResult)
        assert result.num_images == 3
        assert result.config.model_name == "yolo11s.pt"
        assert mock_model.predict.call_count == 3

    def test_raises_on_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No image files found"):
            run_ultralytics_benchmark("yolo11s.pt", tmp_path, num_images=10)


class TestBuildFullMatrix:
    def test_single_model_produces_four_configs(self) -> None:
        configs = build_full_matrix(models=["yolo11s"])

        assert len(configs) == 4

    def test_three_models_produce_twelve_configs(self) -> None:
        configs = build_full_matrix(models=["yolo11s", "yolo26s", "yolo12s"])

        assert len(configs) == 12

    def test_all_backend_preprocess_combinations(self) -> None:
        configs = build_full_matrix(models=["yolo11s"])

        combos = {(c.backend, c.preprocess_mode) for c in configs}
        assert combos == {
            ("onnx", "raw"),
            ("onnx", "dali"),
            ("tensorrt", "raw"),
            ("tensorrt", "dali"),
        }

    def test_custom_parameters_propagated(self) -> None:
        configs = build_full_matrix(
            models=["yolo11s"],
            triton_url="triton:8001",
            target_size=320,
            num_images=50,
            warmup_runs=10,
        )

        for config in configs:
            assert config.triton_url == "triton:8001"
            assert config.target_size == 320
            assert config.num_images == 50
            assert config.warmup_runs == 10

    def test_model_names_preserved(self) -> None:
        models = ["yolo11s", "yolo26s"]
        configs = build_full_matrix(models=models)

        model_names = {c.model_name for c in configs}
        assert model_names == {"yolo11s", "yolo26s"}

    def test_empty_models_returns_empty(self) -> None:
        configs = build_full_matrix(models=[])

        assert configs == []

    def test_returns_benchmark_config_instances(self) -> None:
        configs = build_full_matrix(models=["yolo11s"])

        for config in configs:
            assert isinstance(config, BenchmarkConfig)
