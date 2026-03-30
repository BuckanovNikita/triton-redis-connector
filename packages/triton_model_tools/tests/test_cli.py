"""Tests for CLI argument parsing and subcommand structure."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from triton_model_tools.__main__ import _build_export_parser, main


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
            patch("triton_model_tools.exporter.export_all_models", mock_export),
        ):
            main()

        mock_export.assert_called_once_with(
            repository_root=Path(repo_dir).resolve(),
            model_names=None,
        )

    def test_main_handles_value_error(self, tmp_path: Path) -> None:
        repo_dir = str(tmp_path / "models")
        mock_export = MagicMock(side_effect=ValueError("bad model"))
        with (
            patch("sys.argv", ["triton-model-tools", "export", "-r", repo_dir]),
            patch("triton_model_tools.exporter.export_all_models", mock_export),
            pytest.raises(SystemExit, match="1"),
        ):
            main()
