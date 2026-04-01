"""PreprocessingProvider — depends on ConfigProvider.

Provides OpenCVDecoder and LetterBoxPreprocessor at APP scope.
"""

from __future__ import annotations

from dishka import Provider, provide

from triton_redis_connector.config.schema import PreprocessingConfig  # noqa: TC001
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder
from triton_redis_connector.preprocessing.letterbox import LetterBoxPreprocessor
from triton_redis_connector.providers.scopes import ConnectorScope


class PreprocessingProvider(Provider):
    """Provides preprocessing objects at APP scope."""

    scope = ConnectorScope.APP

    @provide
    def opencv_decoder(self) -> OpenCVDecoder:  # noqa: D102
        return OpenCVDecoder()

    @provide
    def letterbox_preprocessor(self, config: PreprocessingConfig) -> LetterBoxPreprocessor:  # noqa: D102
        return LetterBoxPreprocessor(target_size=config.default_target_size)
