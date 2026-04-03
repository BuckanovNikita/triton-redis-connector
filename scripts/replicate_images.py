"""Replicate (tile) images to fill a target resolution for benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def main() -> None:
    """Tile each image in input-dir to target resolution and write to output-dir."""
    parser = argparse.ArgumentParser(description="Tile images to target resolution")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    for path in tqdm(image_paths, desc="Replicating", unit="img"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        nx = -(-args.target_width // w)
        ny = -(-args.target_height // h)
        tiled: np.ndarray[tuple[int, ...], np.dtype[np.uint8]] = np.tile(img, (ny, nx, 1))
        cropped = tiled[: args.target_height, : args.target_width]
        cv2.imwrite(str(args.output_dir / path.name), cropped)


if __name__ == "__main__":
    main()
