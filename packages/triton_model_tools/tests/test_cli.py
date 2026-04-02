"""Tests for CLI argument parsing and subcommand structure."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from triton_model_tools.__main__ import main

if TYPE_CHECKING:
    from pathlib import Path


class TestMainEntryPoint:
    def test_main_no_args_exits_with_code_1(self) -> None:
        with patch("sys.argv", ["triton-model-tools"]), pytest.raises(SystemExit, match="1"):
            main()

    def test_main_handles_value_error(self, tmp_path: Path) -> None:
        with (
            patch("sys.argv", ["triton-model-tools", "infer", "--image-dir", str(tmp_path), "-m", "yolo11s"]),
            patch(
                "triton_model_tools.inference.folder_inference.run_folder_inference",
                side_effect=ValueError("bad model"),
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

    def test_main_handles_keyboard_interrupt(self, tmp_path: Path) -> None:
        with (
            patch("sys.argv", ["triton-model-tools", "infer", "--image-dir", str(tmp_path), "-m", "yolo11s"]),
            patch(
                "triton_model_tools.inference.folder_inference.run_folder_inference",
                side_effect=KeyboardInterrupt,
            ),
            pytest.raises(SystemExit, match="130"),
        ):
            main()
