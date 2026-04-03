#!/usr/bin/env bash
set -euo pipefail

# PostToolUse hook: auto-run ruff format + ruff check --fix on modified Python files
# Stdin: JSON with tool_input.file_path

INPUT=$(cat)

exec /usr/bin/python3 -c '
import json, sys, subprocess, os, hashlib

data = json.loads(sys.argv[1])
fpath = data.get("tool_input", {}).get("file_path", "")

if not fpath or not fpath.endswith(".py"):
    sys.exit(0)

if not os.path.isfile(fpath):
    sys.exit(0)

project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")

with open(fpath, "rb") as f:
    before = hashlib.md5(f.read()).hexdigest()

subprocess.run(
    ["uv", "run", "ruff", "format", fpath],
    cwd=project_dir, capture_output=True,
)
subprocess.run(
    ["uv", "run", "ruff", "check", "--fix", fpath],
    cwd=project_dir, capture_output=True,
)

with open(fpath, "rb") as f:
    after = hashlib.md5(f.read()).hexdigest()

if before != after:
    result = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"Auto-formatted {fpath} (ruff format + check --fix). File on disk updated.",
        }
    }
    json.dump(result, sys.stdout)
' "$INPUT"
