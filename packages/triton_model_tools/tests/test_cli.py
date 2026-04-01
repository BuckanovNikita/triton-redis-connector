"""Tests for CLI argument parsing and subcommand structure."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from triton_model_tools.__main__ import main
from triton_model_tools.cli.full_benchmark import build_parser as build_full_benchmark_parser


class TestMainEntryPoint:
    def test_main_no_args_exits_with_code_1(self) -> None:
        with patch("sys.argv", ["triton-model-tools"]), pytest.raises(SystemExit, match="1"):
            main()

    def test_main_handles_value_error(self, tmp_path: Path) -> None:
        with (
            patch("sys.argv", ["triton-model-tools", "validate", "-r", str(tmp_path)]),
            patch("triton_model_tools.export.validate.validate_repository", side_effect=ValueError("bad model")),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

    def test_main_handles_keyboard_interrupt(self, tmp_path: Path) -> None:
        with (
            patch("sys.argv", ["triton-model-tools", "validate", "-r", str(tmp_path)]),
            patch(
                "triton_model_tools.export.validate.validate_repository",
                side_effect=KeyboardInterrupt,
            ),
            pytest.raises(SystemExit, match="130"),
        ):
            main()


class TestFullBenchmarkSubcommandParsing:
    def _parse(self, args: list[str]) -> __import__("argparse").Namespace:
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_full_benchmark_parser(subparsers)
        return parser.parse_args(args)

    def test_requires_image_dir(self) -> None:
        with pytest.raises(SystemExit, match="2"):
            self._parse(["full-benchmark", "--models", "yolo11s", "-o", "out.json"])

    def test_models_default_none(self, tmp_path: Path) -> None:
        ns = self._parse(["full-benchmark", "--image-dir", str(tmp_path), "-o", "out.json"])
        assert ns.models is None

    def test_requires_output(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="2"):
            self._parse(["full-benchmark", "--image-dir", str(tmp_path), "--models", "yolo11s"])

    def test_parses_all_arguments(self, tmp_path: Path) -> None:
        ns = self._parse(
            [
                "full-benchmark",
                "--image-dir",
                str(tmp_path),
                "--models",
                "yolo11s",
                "yolo26s",
                "--triton-url",
                "triton:8001",
                "--target-size",
                "320",
                "--num-images",
                "50",
                "--warmup-runs",
                "3",
                "-o",
                "results.json",
            ]
        )

        assert ns.command == "full-benchmark"
        assert ns.image_dir == tmp_path
        assert ns.models == ["yolo11s", "yolo26s"]
        assert ns.triton_url == "triton:8001"
        assert ns.target_size == 320
        assert ns.num_images == 50
        assert ns.warmup_runs == 3
        assert ns.output == Path("results.json")

    def test_default_values(self, tmp_path: Path) -> None:
        ns = self._parse(
            [
                "full-benchmark",
                "--image-dir",
                str(tmp_path),
                "--models",
                "yolo11s",
                "-o",
                "out.json",
            ]
        )

        assert ns.triton_url == "localhost:8001"
        assert ns.target_size == 640
        assert ns.num_images == 100
        assert ns.warmup_runs == 5

    def test_main_dispatches_full_benchmark(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(3):
            (image_dir / f"img{i}.jpg").write_bytes(b"fake")

        output_path = tmp_path / "output.json"

        mock_suite = MagicMock(results=[])
        mock_build = MagicMock(return_value=[])
        mock_run = MagicMock(return_value=mock_suite)
        mock_save = MagicMock()

        with (
            patch(
                "sys.argv",
                [
                    "triton-model-tools",
                    "full-benchmark",
                    "--image-dir",
                    str(image_dir),
                    "--models",
                    "yolo11s",
                    "-o",
                    str(output_path),
                ],
            ),
            patch("triton_model_tools.benchmarks.benchmark.build_full_matrix", mock_build),
            patch("triton_model_tools.benchmarks.benchmark.run_benchmark_matrix", mock_run),
            patch("triton_model_tools.benchmarks.benchmark.save_benchmark_output", mock_save),
        ):
            main()

        mock_build.assert_called_once()
        mock_run.assert_called_once()
        mock_save.assert_called_once()
