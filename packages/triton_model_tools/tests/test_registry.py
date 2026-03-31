"""Tests for YoloModelVariant image size validation and resolved_imgsz."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from triton_model_tools.models import YoloModelVariant


class TestYoloModelVariantImgsz:
    def test_int_imgsz_default(self) -> None:
        variant = YoloModelVariant(name="test", ultralytics_model="test.pt")
        assert variant.imgsz == 640
        assert variant.resolved_imgsz == (640, 640)

    def test_int_imgsz_custom(self) -> None:
        variant = YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=320)
        assert variant.imgsz == 320
        assert variant.resolved_imgsz == (320, 320)

    def test_tuple_imgsz(self) -> None:
        variant = YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=(960, 640))
        assert variant.imgsz == (960, 640)
        assert variant.resolved_imgsz == (960, 640)

    def test_tuple_imgsz_nonsquare(self) -> None:
        variant = YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=(384, 352))
        assert variant.imgsz == (384, 352)
        assert variant.resolved_imgsz == (384, 352)

    def test_rejects_zero_int(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=0)

    def test_rejects_negative_int(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=-640)

    def test_rejects_not_divisible_by_32_int(self) -> None:
        with pytest.raises(ValidationError, match="not divisible by 32"):
            YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=641)

    def test_rejects_tuple_with_zero(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=(0, 640))

    def test_rejects_tuple_not_divisible_by_32(self) -> None:
        with pytest.raises(ValidationError, match="not divisible by 32"):
            YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=(600, 601))

    def test_rejects_tuple_with_negative(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            YoloModelVariant(name="test", ultralytics_model="test.pt", imgsz=(-1, 640))
