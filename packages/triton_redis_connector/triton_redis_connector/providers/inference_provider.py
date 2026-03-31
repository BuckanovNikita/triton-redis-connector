"""InferenceProvider — depends on ConfigProvider.

Provides InferenceWorkerFactory at APP scope. The factory creates
per-worker tritonclient instances when ProcessPoolProcessor spawns
worker processes.
"""

from __future__ import annotations

from dishka import Provider, provide

from triton_redis_connector.config.schema import InferenceConfig  # noqa: TC001 — dishka needs runtime access
from triton_redis_connector.inference.worker import InferenceWorkerFactory
from triton_redis_connector.providers.scopes import ConnectorScope


class InferenceProvider(Provider):
    """Provides inference-related objects at APP scope."""

    scope = ConnectorScope.APP

    @provide
    def inference_worker_factory(self, config: InferenceConfig) -> InferenceWorkerFactory:  # noqa: D102
        return InferenceWorkerFactory(config=config)
