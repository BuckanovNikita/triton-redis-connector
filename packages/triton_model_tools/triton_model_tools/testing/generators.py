"""Concrete StreamGenerator implementations for testing and load generation."""

import itertools
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from triton_model_tools.testing.stream_populator import ImageStreamEntry, InspectionStreamEntry

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})

_DEFAULT_METAL_TYPES = ("steel", "copper", "aluminum")


def _collect_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        msg = f"Images directory does not exist: {images_dir}"
        raise FileNotFoundError(msg)
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not images:
        msg = f"No images found in {images_dir}"
        raise FileNotFoundError(msg)
    return images


class ImageStreamGenerator:
    """Generates ImageStreamEntry events by reading image files from a directory.

    Cycles through images indefinitely, assigning unique request IDs.
    Used by e2e tests and benchmarks.
    """

    event_type: type[BaseModel] = ImageStreamEntry

    def __init__(  # noqa: D107
        self,
        images_dir: Path,
        model_name: str = "",
        stream_name: str = "camera_images",
    ) -> None:
        self._images = _collect_images(images_dir)
        self._model_name = model_name
        self.stream_name = stream_name

    def __call__(self) -> Iterator[ImageStreamEntry]:
        """Yield ImageStreamEntry events indefinitely, cycling through images."""
        for image_path in itertools.cycle(self._images):
            yield ImageStreamEntry(
                data=image_path.read_bytes(),
                request_id=uuid.uuid4().hex,
                model_name=self._model_name,
                image_name=image_path.name,
            )


class InspectionStreamConfig(BaseModel):
    """Configuration for InspectionStreamGenerator to avoid too many __init__ params."""

    model_config = ConfigDict(frozen=True)

    camera_count: int = 3
    metal_types: tuple[str, ...] = _DEFAULT_METAL_TYPES
    event_type_label: str = "image"
    stream_name: str = "camera_images"
    max_count: int | None = None


class InspectionStreamGenerator:
    """Generates InspectionStreamEntry events with cycling camera IDs and metal types.

    Used by the CLI e2e-test populate-only mode.
    """

    event_type: type[BaseModel] = InspectionStreamEntry

    def __init__(  # noqa: D107
        self,
        images_dir: Path,
        config: InspectionStreamConfig | None = None,
    ) -> None:
        cfg = config or InspectionStreamConfig()
        all_images = _collect_images(images_dir)
        self._images = all_images[: cfg.max_count] if cfg.max_count is not None else all_images
        self._camera_count = cfg.camera_count
        self._metal_types = cfg.metal_types
        self._event_type_label = cfg.event_type_label
        self.stream_name = cfg.stream_name

    def __call__(self) -> Iterator[InspectionStreamEntry]:
        """Yield InspectionStreamEntry events indefinitely, cycling through images."""
        for idx, image_path in enumerate(itertools.cycle(self._images)):
            yield InspectionStreamEntry(
                camera_no=(idx % self._camera_count) + 1,
                image_path=str(image_path.resolve()),
                frame_counter=idx + 1,
                receipt_ts=time.time(),
                metal_type=self._metal_types[idx % len(self._metal_types)],
                event_type=self._event_type_label,
            )


class CameraEventEntry(BaseModel):
    """Camera event with cam_id and image_path for the load generator."""

    cam_id: str
    image_path: str


class CameraEventGenerator:
    """Generates camera events by cycling through images and camera IDs.

    Replaces reference_implementation/load_generator.py logic.
    """

    event_type: type[BaseModel] = CameraEventEntry

    def __init__(  # noqa: D107
        self,
        images_dir: Path,
        num_cams: int = 6,
        stream_name: str = "camera:input",
    ) -> None:
        self._images = _collect_images(images_dir)
        self._num_cams = num_cams
        self.stream_name = stream_name

    def __call__(self) -> Iterator[CameraEventEntry]:
        """Yield CameraEventEntry events indefinitely, cycling through images and camera IDs."""
        image_cycle = itertools.cycle(self._images)
        cam_cycle = itertools.cycle(range(1, self._num_cams + 1))
        while True:
            image = next(image_cycle)
            cam_id = next(cam_cycle)
            yield CameraEventEntry(
                cam_id=str(cam_id),
                image_path=str(image.resolve()),
            )
