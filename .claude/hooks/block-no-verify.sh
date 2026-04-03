#!/usr/bin/env bash
set -euo pipefail

# PreToolUse hook: block git commit --no-verify
# Stdin: JSON with tool_input.command

echo "Fix failing checks instead of skipping them with --no-verify" >&2
exit 2
