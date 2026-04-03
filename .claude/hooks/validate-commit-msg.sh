#!/usr/bin/env bash
set -euo pipefail

# PreToolUse hook: validate conventional commit message format
# Handles both -m <msg> and -F <file> commit styles
# Stdin: JSON with tool_input.command

INPUT=$(cat)

exec /usr/bin/python3 -c '
import json, sys, re

data = json.loads(sys.argv[1])
cmd = data.get("tool_input", {}).get("command", "")

if "--amend" in cmd and "-m" not in cmd and "-F" not in cmd:
    sys.exit(0)

if "--allow-empty-message" in cmd:
    sys.exit(0)

msg = ""

m = re.search(r"-F\s+(\S+)", cmd)
if m:
    fpath = m.group(1)
    try:
        with open(fpath) as f:
            msg = f.readline().strip()
    except (FileNotFoundError, PermissionError):
        sys.exit(0)

if not msg:
    m = re.search(r"-m\s+((?:\\\s|\S)+)", cmd)
    if m:
        msg = m.group(1).replace("\\ ", " ")

if not msg:
    sys.exit(0)

PATTERN = r"^(feat|fix|refactor|test|docs|chore|perf|ci|build|style|revert)(\(.+\))?!?:\s+.+"
if re.match(PATTERN, msg):
    sys.exit(0)

print(
    f"Commit message must use Conventional Commits format "
    f"(feat:|fix:|refactor:|test:|docs:|chore:). Got: {msg}",
    file=sys.stderr,
)
sys.exit(2)
' "$INPUT"
