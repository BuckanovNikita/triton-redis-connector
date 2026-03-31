"""Shared type definitions for CLI subcommand modules."""

from __future__ import annotations

import argparse
from typing import Any, Protocol

_YOLO_STRIDE = 32


class SubParsers(Protocol):
    """Protocol for argparse subparsers action (avoids private _SubParsersAction)."""

    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:  # noqa: ANN401
        ...


def parse_image_size(s: str) -> int | tuple[int, int]:
    """Parse an image size string: '640' -> 640, '960x600' -> (960, 600)."""
    if "x" in s:
        parts = s.split("x")
        if len(parts) != 2:
            msg = f"Invalid image size format: {s!r}. Expected 'SIZE' or 'HEIGHTxWIDTH'"
            raise argparse.ArgumentTypeError(msg)
        h, w = int(parts[0]), int(parts[1])
        for dim in (h, w):
            if dim <= 0:
                msg = f"Image dimension {dim} must be positive"
                raise argparse.ArgumentTypeError(msg)
            if dim % _YOLO_STRIDE != 0:
                msg = f"Image dimension {dim} is not divisible by 32 (YOLO stride requirement)"
                raise argparse.ArgumentTypeError(msg)
        return (h, w)
    val = int(s)
    if val <= 0:
        msg = f"Image dimension {val} must be positive"
        raise argparse.ArgumentTypeError(msg)
    if val % _YOLO_STRIDE != 0:
        msg = f"Image dimension {val} is not divisible by 32 (YOLO stride requirement)"
        raise argparse.ArgumentTypeError(msg)
    return val
