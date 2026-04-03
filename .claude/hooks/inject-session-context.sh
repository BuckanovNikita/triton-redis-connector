#!/usr/bin/env bash
set -euo pipefail

# UserPromptSubmit hook (once): inject project context at session start
# Provides Triton version, git branch, dirty state

cd "${CLAUDE_PROJECT_DIR:-.}"

TRITON_VERSION=""
if [ -f .env ]; then
    TRITON_VERSION=$(grep -E "^TRITON_VERSION=" .env 2>/dev/null | cut -d= -f2 || true)
fi

BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

CONTEXT="Session context: branch=$BRANCH, dirty_files=$DIRTY"
if [ -n "$TRITON_VERSION" ]; then
    CONTEXT="$CONTEXT, TRITON_VERSION=$TRITON_VERSION"
fi

/usr/bin/python3 -c '
import json, sys
ctx = sys.argv[1]
result = {
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": ctx,
    }
}
json.dump(result, sys.stdout)
' "$CONTEXT"
