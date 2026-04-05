#!/usr/bin/env bash
set -euo pipefail

# PreToolUse hook: block git commit --no-verify
# Stdin: JSON with tool_input.command

INPUT=$(cat)
CMD=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

if echo "$CMD" | grep -qE 'git\s+commit.*--no-verify'; then
  echo "Fix failing checks instead of skipping them with --no-verify" >&2
  exit 2
fi

exit 0
