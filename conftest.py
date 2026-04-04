"""Root conftest providing shared test fixtures across all packages."""

from pathlib import Path

import pytest

TEST_FIXTURES = Path(__file__).resolve().parent / "test_fixtures"
_RESTORE_HINT = "Run: uv run python test_fixtures/restore_models.py"


@pytest.fixture
def test_images_dir() -> Path:
    """Return path to synthetic test images; skip if unavailable."""
    path = TEST_FIXTURES / "images"
    if not path.is_dir() or not any(path.glob("*.jpg")):
        pytest.skip("Test images not found. Run: uv run python test_fixtures/create_test_images.py")
    return path


@pytest.fixture
def test_models_dir() -> Path:
    """Return path to test ONNX models; skip if unavailable."""
    path = TEST_FIXTURES / "models"
    if not path.is_dir() or not any(path.rglob("*.onnx")):
        pytest.skip(f"Models not found. {_RESTORE_HINT}")
    return path


@pytest.fixture
def corrupted_models_dir() -> Path:
    """Return path to corrupted test models; skip if unavailable."""
    path = TEST_FIXTURES / "corrupted_models"
    if not path.is_dir() or not any(path.rglob("*.onnx")):
        pytest.skip(f"Corrupted models not found. {_RESTORE_HINT}")
    return path
