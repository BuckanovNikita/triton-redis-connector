#!/usr/bin/env bash
set -uo pipefail

# PreToolUse hook: run pre-commit before allowing git commit
# Blocks commit, runs pre-commit, blocks if it fails
# Stdin: JSON with tool_input.command

INPUT=$(cat)

CMD=$(/usr/bin/python3 -c '
import json, sys
data = json.loads(sys.argv[1])
print(data.get("tool_input", {}).get("command", ""))
' "$INPUT")

# Skip if --no-verify (handled by block-no-verify.sh)
case "$CMD" in
    *--no-verify*) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-.}"

# Run pre-commit and check exit code directly
uv run pre-commit run --all-files
RC=$?

if [ "$RC" -ne 0 ]; then
    echo "pre-commit failed (exit code $RC). Fix issues before committing." >&2
    exit 2
fi

exit 0
