"""Tests for COCO dataset downloader."""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from triton_model_tools.dataset import (
    COCO_VAL2017_DIR_NAME,
    download_coco_val,
)

if TYPE_CHECKING:
    from pathlib import Path


def _create_fake_coco_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(10):
            zf.writestr(f"val2017/image_{i:04d}.jpg", f"fake_image_data_{i}")
    return buf.getvalue()


class TestDownloadCocoVal:
    def test_skips_when_directory_exists_with_images(self, tmp_path: Path) -> None:
        val_dir = tmp_path / COCO_VAL2017_DIR_NAME
        val_dir.mkdir()
        (val_dir / "existing.jpg").write_bytes(b"existing_image")

        result = download_coco_val(tmp_path)

        assert result == val_dir

    @patch("triton_model_tools.dataset.urllib.request.urlopen")
    def test_downloads_and_extracts(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        zip_data = _create_fake_coco_zip()
        mock_response = MagicMock()
        mock_response.read.side_effect = [zip_data, b""]
        mock_response.headers = {"Content-Length": str(len(zip_data))}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = download_coco_val(tmp_path)

        assert result == tmp_path / COCO_VAL2017_DIR_NAME
        assert result.is_dir()
        extracted_files = list(result.iterdir())
        assert len(extracted_files) == 10

    @patch("triton_model_tools.dataset.urllib.request.urlopen")
    def test_max_images_limits_extraction(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        zip_data = _create_fake_coco_zip()
        mock_response = MagicMock()
        mock_response.read.side_effect = [zip_data, b""]
        mock_response.headers = {"Content-Length": str(len(zip_data))}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = download_coco_val(tmp_path, max_images=3)

        assert result.is_dir()
        extracted_files = list(result.iterdir())
        assert len(extracted_files) == 3

    @patch("triton_model_tools.dataset.urllib.request.urlopen")
    def test_creates_target_dir_if_not_exists(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        zip_data = _create_fake_coco_zip()
        mock_response = MagicMock()
        mock_response.read.side_effect = [zip_data, b""]
        mock_response.headers = {"Content-Length": str(len(zip_data))}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        nested_dir = tmp_path / "nested" / "path"
        result = download_coco_val(nested_dir)

        assert result.is_dir()
        assert nested_dir.is_dir()

    def test_skips_empty_directory(self, tmp_path: Path) -> None:
        val_dir = tmp_path / COCO_VAL2017_DIR_NAME
        val_dir.mkdir()

        with patch("triton_model_tools.dataset.urllib.request.urlopen") as mock_urlopen:
            zip_data = _create_fake_coco_zip()
            mock_response = MagicMock()
            mock_response.read.side_effect = [zip_data, b""]
            mock_response.headers = {"Content-Length": str(len(zip_data))}
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = download_coco_val(tmp_path)

            mock_urlopen.assert_called_once()
            assert result == val_dir
