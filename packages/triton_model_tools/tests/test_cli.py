"""Tests for CLI argument parsing and subcommand structure."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from triton_model_tools.__main__ import _build_export_parser, _build_full_benchmark_parser, main


@pytest.fixture
def repo_path(tmp_path: Path) -> str:
    return str(tmp_path / "model_repository")


class TestExportSubcommandParsing:
    def _parse(self, args: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        _build_export_parser(subparsers)
        return parser.parse_args(args)

    def test_export_requires_repository_root(self) -> None:
        with pytest.raises(SystemExit, match="2"):
            self._parse(["export"])

    def test_export_with_repository_root(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path])
        assert ns.repository_root == Path(repo_path)
        assert ns.command == "export"

    def test_export_format_default_is_onnx(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path])
        assert ns.export_format == "onnx"

    def test_export_format_onnx(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path, "-f", "onnx"])
        assert ns.export_format == "onnx"

    def test_export_format_tensorrt(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path, "-f", "tensorrt"])
        assert ns.export_format == "tensorrt"

    def test_export_format_all(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path, "-f", "all"])
        assert ns.export_format == "all"

    def test_export_format_rejects_invalid(self, repo_path: str) -> None:
        with pytest.raises(SystemExit, match="2"):
            self._parse(["export", "-r", repo_path, "-f", "invalid"])

    def test_dali_ensemble_flag_default_false(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path])
        assert ns.dali_ensemble is False

    def test_dali_ensemble_flag_long(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path, "--dali-ensemble"])
        assert ns.dali_ensemble is True

    def test_dali_ensemble_flag_short(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path, "--dali"])
        assert ns.dali_ensemble is True

    def test_nms_flag_default_false(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path])
        assert ns.nms is False

    def test_nms_flag(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path, "--nms"])
        assert ns.nms is True

    def test_models_default_none(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path])
        assert ns.models is None

    def test_models_single(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path, "-m", "yolo11s"])
        assert ns.models == ["yolo11s"]

    def test_models_multiple(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path, "-m", "yolo11s", "yolo26s"])
        assert ns.models == ["yolo11s", "yolo26s"]

    def test_all_flags_combined(self, repo_path: str) -> None:
        ns = self._parse(
            [
                "export",
                "-r",
                repo_path,
                "-f",
                "all",
                "--dali-ensemble",
                "--nms",
                "-m",
                "yolo11s",
            ]
        )
        assert ns.repository_root == Path(repo_path)
        assert ns.export_format == "all"
        assert ns.dali_ensemble is True
        assert ns.nms is True
        assert ns.models == ["yolo11s"]

    def test_func_is_set_on_export(self, repo_path: str) -> None:
        ns = self._parse(["export", "-r", repo_path])
        assert callable(ns.func)


class TestMainEntryPoint:
    def test_main_is_callable(self) -> None:
        assert callable(main)

    def test_main_no_args_exits_with_code_1(self) -> None:
        with patch("sys.argv", ["triton-model-tools"]), pytest.raises(SystemExit, match="1"):
            main()

    def test_main_dispatches_export_subcommand(self, tmp_path: Path) -> None:
        repo_dir = str(tmp_path / "models")
        mock_export = MagicMock(return_value=[tmp_path / "models" / "yolo11s"])
        with (
            patch("sys.argv", ["triton-model-tools", "export", "-r", repo_dir]),
            patch("triton_model_tools.export.exporter.export_all_models", mock_export),
        ):
            main()

        mock_export.assert_called_once_with(
            repository_root=Path(repo_dir).resolve(),
            model_names=None,
            fmt="onnx",
        )

    def test_main_handles_value_error(self, tmp_path: Path) -> None:
        repo_dir = str(tmp_path / "models")
        mock_export = MagicMock(side_effect=ValueError("bad model"))
        with (
            patch("sys.argv", ["triton-model-tools", "export", "-r", repo_dir]),
            patch("triton_model_tools.export.exporter.export_all_models", mock_export),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

    def test_main_handles_keyboard_interrupt(self, tmp_path: Path) -> None:
        repo_dir = str(tmp_path / "models")
        mock_export = MagicMock(side_effect=KeyboardInterrupt)
        with (
            patch("sys.argv", ["triton-model-tools", "export", "-r", repo_dir]),
            patch("triton_model_tools.export.exporter.export_all_models", mock_export),
            pytest.raises(SystemExit, match="130"),
        ):
            main()

    def test_run_export_with_explicit_models(self, tmp_path: Path) -> None:
        repo_dir = str(tmp_path / "models")
        mock_export = MagicMock(return_value=[tmp_path / "models" / "yolo11s"])
        with (
            patch("sys.argv", ["triton-model-tools", "export", "-r", repo_dir, "-m", "yolo11s"]),
            patch("triton_model_tools.export.exporter.export_all_models", mock_export),
        ):
            main()

        mock_export.assert_called_once_with(
            repository_root=Path(repo_dir).resolve(),
            model_names=["yolo11s"],
            fmt="onnx",
        )

    def test_run_export_with_dali_ensemble(self, tmp_path: Path) -> None:
        repo_dir = str(tmp_path / "models")
        model_dir = tmp_path / "models" / "yolo11s"
        model_dir.mkdir(parents=True)
        mock_export = MagicMock(return_value=[model_dir])
        mock_build_ensemble = MagicMock()
        with (
            patch("sys.argv", ["triton-model-tools", "export", "-r", repo_dir, "--dali-ensemble"]),
            patch("triton_model_tools.export.exporter.export_all_models", mock_export),
            patch("triton_model_tools.export.dali.build_ensemble_directory", mock_build_ensemble),
        ):
            main()

        mock_build_ensemble.assert_called_once()
        call_kwargs = mock_build_ensemble.call_args
        assert call_kwargs.kwargs["model_name"] == "yolo11s"

    def test_run_export_with_nms(self, tmp_path: Path) -> None:
        repo_dir = str(tmp_path / "models")
        mock_export = MagicMock(return_value=[tmp_path / "models" / "yolo11s"])
        mock_build_nms = MagicMock()
        with (
            patch("sys.argv", ["triton-model-tools", "export", "-r", repo_dir, "--nms"]),
            patch("triton_model_tools.export.exporter.export_all_models", mock_export),
            patch("triton_model_tools.export.nms.build_nms_model_directory", mock_build_nms),
        ):
            main()

        mock_build_nms.assert_called_once()

    def test_run_export_dali_nms_skips_end2end(self, tmp_path: Path) -> None:
        repo_dir = str(tmp_path / "models")
        e2e_dir = tmp_path / "models" / "yolo26s"
        e2e_dir.mkdir(parents=True)
        std_dir = tmp_path / "models" / "yolo11s"
        std_dir.mkdir(parents=True)
        mock_export = MagicMock(return_value=[e2e_dir, std_dir])
        mock_build_ensemble = MagicMock()
        mock_build_nms = MagicMock()
        with (
            patch(
                "sys.argv",
                ["triton-model-tools", "export", "-r", repo_dir, "--dali-ensemble", "--nms"],
            ),
            patch("triton_model_tools.export.exporter.export_all_models", mock_export),
            patch("triton_model_tools.export.dali.build_ensemble_directory", mock_build_ensemble),
            patch("triton_model_tools.export.nms.build_nms_model_directory", mock_build_nms),
        ):
            main()

        assert mock_build_ensemble.call_count == 2
        calls = mock_build_ensemble.call_args_list
        e2e_call = next(c for c in calls if c.kwargs["model_name"] == "yolo26s")
        std_call = next(c for c in calls if c.kwargs["model_name"] == "yolo11s")
        assert e2e_call.kwargs["nms_model_name"] is None
        assert std_call.kwargs["nms_model_name"] == "nms_postprocess"


class TestFullBenchmarkSubcommandParsing:
    def _parse(self, args: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        _build_full_benchmark_parser(subparsers)
        return parser.parse_args(args)

    def test_requires_image_dir(self) -> None:
        with pytest.raises(SystemExit, match="2"):
            self._parse(["full-benchmark", "--models", "yolo11s", "-o", "out.json"])

    def test_requires_models(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="2"):
            self._parse(["full-benchmark", "--image-dir", str(tmp_path), "-o", "out.json"])

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

    def test_func_is_set(self, tmp_path: Path) -> None:
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

        assert callable(ns.func)

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
