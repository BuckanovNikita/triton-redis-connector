"""Check infrastructure-independent imports in the repaired acceptance tests."""

# Standalone acceptance script, not an importable package.
# ruff: noqa: INP001

import ast
from pathlib import Path

FORBIDDEN = frozenset(
    {"triton_redis_connector", "triton_model_tools", "triton_shared_models", "torch", "torchvision", "fiftyone"}
)
TEST_PATHS = (
    "packages/typed_flow/tests/test_engine/test_app.py",
    "packages/typed_flow/tests/test_metrics/test_emitter.py",
)


def prohibited_imports(source: str) -> set[str]:
    """Return forbidden imported roots, including imports inside functions."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots & FORBIDDEN


def main() -> None:
    """Validate actual imports and prove a prohibited dependency is detected."""
    repo = Path(__file__).resolve().parents[3]
    for relative in (*TEST_PATHS, str(Path(__file__).relative_to(repo))):
        violations = prohibited_imports((repo / relative).read_text())
        if violations:
            raise SystemExit(f"{relative}: forbidden infrastructure imports {sorted(violations)}")
    for forbidden in FORBIDDEN:
        for source in (f"import {forbidden}\n", f"def example():\n    from {forbidden}.adapter import value\n"):
            if prohibited_imports(source) != {forbidden}:
                raise SystemExit(f"Negative import check failed to detect {forbidden}")


if __name__ == "__main__":
    main()
