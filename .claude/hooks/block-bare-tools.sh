#!/usr/bin/env bash
set -euo pipefail

# PreToolUse hook: block bare Python tool invocations (must use uv run)
# Also blocks .venv/bin/ invocations
# Stdin: JSON with tool_input.command

INPUT=$(cat)

exec /usr/bin/python3 -c '
import json, sys, re, os

data = json.loads(sys.argv[1])
cmd = data.get("tool_input", {}).get("command", "").strip()

if not cmd:
    sys.exit(0)

if ".venv/bin/" in cmd:
    print("Use uv run instead of .venv/bin/ -- project convention", file=sys.stderr)
    sys.exit(2)

stripped = re.sub(r"^(\w+=\S+\s+)*", "", cmd)
parts = stripped.split()
if not parts:
    sys.exit(0)

first_word = os.path.basename(parts[0])

BLOCKED = {
    "python", "python3",
    "pip", "pip3",
    "pytest",
    "ruff",
    "mypy",
    "pyright",
    "vulture",
    "lint-imports",
    "pre-commit",
    "prek",
}

if first_word not in BLOCKED:
    sys.exit(0)

if stripped.startswith("uv run "):
    sys.exit(0)

print(
    f"Use uv run {first_word} instead of bare {first_word} -- project convention",
    file=sys.stderr,
)
sys.exit(2)
' "$INPUT"
