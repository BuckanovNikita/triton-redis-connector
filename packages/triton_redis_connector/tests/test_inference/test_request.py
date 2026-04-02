"""Tests for TritonRequest mutual exclusion validator."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from triton_redis_connector.inference import TritonRequest


class TestTritonRequest:
    def test_both_modes_raises(self) -> None:
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        with pytest.raises(ValidationError, match="Exactly one"):
            TritonRequest(
                model_name="yolo11s",
                raw_image=tensor,
                image_bytes=b"\xff\xd8\xff",
            )

    def test_neither_mode_raises(self) -> None:
        with pytest.raises(ValidationError, match="Exactly one"):
            TritonRequest(model_name="yolo11s")
