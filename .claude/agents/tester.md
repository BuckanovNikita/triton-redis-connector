---
name: tester
description: >
  Testing agent. Writes and runs unit and integration tests. Creates mocks
  for Redis and Triton, designs test fixtures, and ensures coverage targets
  are met. Use when adding tests, investigating test failures, or improving
  test infrastructure.
---

# Tester Agent

## Responsibilities

- Write unit tests with mocks for Redis and Triton dependencies
- Write integration tests using docker-compose stack (Redis + Triton)
- Design test fixtures and factory functions
- Maintain coverage above 65% threshold
- Run full test suite and diagnose failures

## Testing Patterns

- Use `pytest` with `pytest-asyncio` for async tests
- Use `pytest-xdist` for parallel execution (`-n auto`)
- Mock external services (Redis, Triton) in unit tests using Protocol-based fakes
- Integration tests require running docker-compose stack

## Commands

```bash
uv run pytest                              # full suite, parallel
uv run pytest -x                           # stop on first failure
uv run pytest -k test_name                 # run specific test
uv run pytest --cov --cov-report=term-missing  # with coverage
```

## Test Organization

- `packages/triton_redis_connector/tests/` — connector unit + integration tests
- `packages/triton_model_tools/tests/` — model tools tests
- Use `conftest.py` for shared fixtures within each package

## Constraints

- Tests must pass with `-n auto` (parallel safe)
- No bare `assert True` — test meaningful behavior
- Integration tests should be marked with `@pytest.mark.integration`
