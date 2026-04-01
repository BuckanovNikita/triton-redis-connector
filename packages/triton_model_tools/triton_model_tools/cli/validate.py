"""Validate a Triton model repository directory structure and config integrity."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``validate`` subcommand."""
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate Triton model repository structure and config.pbtxt files",
    )
    validate_parser.add_argument(
        "-r",
        "--repository-root",
        type=Path,
        required=True,
        help="Path to Triton model repository",
    )
    validate_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``validate`` subcommand."""
    from triton_model_tools.export.validate import validate_repository  # noqa: PLC0415

    repository_root: Path = args.repository_root.resolve()
    logger.info("Validating repository: {}", repository_root)

    results = validate_repository(repository_root)

    has_errors = False
    for errors in results.values():
        if errors:
            has_errors = True
            for error in errors:
                logger.error("{}", error)

    if has_errors:
        logger.error("Repository validation failed")
    else:
        logger.info("Repository validation passed: all {} models valid", len(results))
