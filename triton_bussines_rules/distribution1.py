"""Sample distribution rule for routing inference requests."""

from pathlib import Path


class InputEvent:
    """Input event schema for distribution routing."""

    image_path: Path
    camera_no: int
    meta_info: str
    double_nesteed_meta_info: dict[int, str]
