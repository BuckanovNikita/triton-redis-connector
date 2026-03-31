"""Shared type definitions for CLI subcommand modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import argparse


class SubParsers(Protocol):
    """Protocol for argparse subparsers action (avoids private _SubParsersAction)."""

    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:  # noqa: ANN401
        ...
