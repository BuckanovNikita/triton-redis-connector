"""Create synthetic test images for testing.

Generates random images with geometric shapes to simulate objects.
Images are saved as JPEG files in the images/ subdirectory.

Run: uv run python test_fixtures/create_test_images.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_test_images(output_dir: Path, count: int = 8) -> None:
    """Generate synthetic test images with random rectangles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(count):
        h = int(rng.integers(480, 1080))
        w = int(rng.integers(640, 1920))
        image = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

        num_rects = int(rng.integers(1, 5))
        for _ in range(num_rects):
            x1 = int(rng.integers(0, w // 2))
            y1 = int(rng.integers(0, h // 2))
            x2 = int(rng.integers(w // 2, w))
            y2 = int(rng.integers(h // 2, h))
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

        filename = f"test_{i:04d}.jpg"
        cv2.imwrite(str(output_dir / filename), image)


if __name__ == "__main__":
    create_test_images(Path(__file__).parent / "images")
