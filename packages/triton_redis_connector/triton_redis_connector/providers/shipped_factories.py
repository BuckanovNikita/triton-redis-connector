"""Config-friendly factories for shipped processors.

Lives in the providers layer so it can import from both engine and
preprocessing layers without violating import-linter contracts.
"""

from __future__ import annotations

from triton_redis_connector.engine.shipped_processors import ClientPreprocessor
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder, TurboJPEGDecoder
from triton_redis_connector.preprocessing.letterbox import LetterBoxPreprocessor


class ConfigurableClientPreprocessor(ClientPreprocessor):
    """Config-driven ClientPreprocessor that constructs decoder/preprocessor from params.

    Accepts simple YAML-friendly parameters instead of protocol instances.
    Suitable for config-driven node assembly via ``NodeConfig.class_path``.
    """

    def __init__(  # noqa: D107
        self,
        model_name: str,
        *,
        decoder_type: str = "turbojpeg",
        target_size: tuple[int, int] | list[int] = (640, 640),
        input_name: str = "images",
        output_names: tuple[str, ...] | list[str] = ("output0",),
    ) -> None:
        decoder: TurboJPEGDecoder | OpenCVDecoder
        if decoder_type == "turbojpeg":
            decoder = TurboJPEGDecoder()
        elif decoder_type == "opencv":
            decoder = OpenCVDecoder()
        else:
            msg = f"Unknown decoder_type: {decoder_type!r}, expected 'turbojpeg' or 'opencv'"
            raise ValueError(msg)

        size: tuple[int, int] = (target_size[0], target_size[1]) if isinstance(target_size, list) else target_size
        preprocessor = LetterBoxPreprocessor(target_size=size)

        super().__init__(
            decoder=decoder,
            preprocessor=preprocessor,
            model_name=model_name,
            input_name=input_name,
            output_names=tuple(output_names),
        )
