"""Dishka dependency injection support for typed-flow pipelines."""

from __future__ import annotations

from typed_flow.di.container import make_app_container
from typed_flow.di.provider import TypedFlowProvider

__all__ = [
    "TypedFlowProvider",
    "make_app_container",
]
