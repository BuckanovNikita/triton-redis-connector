#!/usr/bin/env bash
set -euo pipefail

# WorktreeCreate hook: set up environment for worktree agents
# Copies .env and syncs dependencies so agents can run immediately
# Stdin: JSON with worktree path info

INPUT=$(cat)

WORKTREE_PATH=$(/usr/bin/python3 -c '
import json, sys
d = json.loads(sys.argv[1])
print(d.get("worktree_path", d.get("cwd", "")))
' "$INPUT" 2>/dev/null || true)

if [ -z "$WORKTREE_PATH" ]; then
    exit 0
fi

MAIN_ENV="${CLAUDE_PROJECT_DIR:-.}/.env"
if [ -f "$MAIN_ENV" ] && [ ! -f "$WORKTREE_PATH/.env" ]; then
    cp "$MAIN_ENV" "$WORKTREE_PATH/.env"
fi

cd "$WORKTREE_PATH"
uv sync --all-groups 2>/dev/null || true

exit 0
