"""Container factory for typed-flow applications."""

from __future__ import annotations

from dishka import Container, make_container

from typed_flow.config.schema import AppConfig
from typed_flow.di.provider import TypedFlowProvider


def make_app_container(config: AppConfig) -> Container:
    """Build a dishka container pre-loaded with typed-flow dependencies."""
    return make_container(TypedFlowProvider(), context={AppConfig: config})
