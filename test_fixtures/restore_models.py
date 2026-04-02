"""Restore test model weights (not committed to git).

Run: uv run python test_fixtures/restore_models.py
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

FIXTURES_DIR = Path(__file__).parent
MODELS_DIR = FIXTURES_DIR / "models"
CORRUPTED_DIR = FIXTURES_DIR / "corrupted_models"


def restore_models() -> None:
    """Create model directories and log restoration status."""
    logger.info("Restoring test models to {}", MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)
    logger.warning("Model restoration not yet implemented. Models must be exported manually.")


if __name__ == "__main__":
    restore_models()
