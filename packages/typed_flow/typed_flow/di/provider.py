"""Dishka provider for typed-flow pipeline dependencies."""

from dishka import Provider, Scope, from_context, provide
from redis import Redis

from typed_flow.config.schema import AppConfig, RedisConnectionConfig


class TypedFlowProvider(Provider):
    """Provides shared Redis client from AppConfig."""

    scope = Scope.APP

    config = from_context(provides=AppConfig, scope=Scope.APP)

    @provide
    def redis_connection_config(self, config: AppConfig) -> RedisConnectionConfig:
        """Extract Redis connection config from the application config."""
        return config.redis

    @provide
    def redis_client(self, cfg: RedisConnectionConfig) -> "Redis[bytes]":
        """Create a shared thread-safe Redis client from connection config."""
        return Redis(host=cfg.host, port=cfg.port, db=cfg.db, decode_responses=False)
