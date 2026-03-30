"""Envelope re-export from protocols layer.

The canonical Envelope definition lives in triton_redis_connector.protocols
(the leaf layer with zero internal dependencies). This module re-exports
it so that engine-layer code can import from engine.event as expected
by the package structure in design.md.
"""

from __future__ import annotations

from triton_redis_connector.protocols import Envelope

__all__ = ["Envelope"]
