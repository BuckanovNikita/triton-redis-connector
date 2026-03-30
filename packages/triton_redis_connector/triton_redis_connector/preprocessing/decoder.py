"""OpenCV-based image decoder implementing ImageDecoderProtocol.

Uses cv2.imdecode which releases the GIL during decode, enabling
concurrent execution in thread pool processors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class OpenCVDecoder:
    """Decodes raw image bytes into BGR HWC uint8 numpy array using OpenCV."""

    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]:
        """Decode raw image bytes into a BGR HWC uint8 numpy array.

        Raises:
            ValueError: If the image cannot be decoded.

        """
        import cv2  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        if not raw:
            msg = "Failed to decode image: empty input bytes"
            raise ValueError(msg)

        buf = np.frombuffer(raw, dtype=np.uint8)
        try:
            image: npt.NDArray[np.uint8] | None = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # type: ignore[assignment]  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]
        except cv2.error as exc:  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
            msg = f"Failed to decode image: {exc}"
            raise ValueError(msg) from exc

        if image is None:
            msg = "Failed to decode image: cv2.imdecode returned None"
            raise ValueError(msg)

        return image
