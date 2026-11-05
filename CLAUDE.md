# CLAUDE.md

@AGENTS.md

## Claude Code adapter

### Project skills (`.claude/skills/`)

| Skill | Trigger |
|---|---|
| `/commit` | Commit and push workflow with pre-commit |
| `/check-infra` | Check Docker services health (Redis, Triton) |
| `/scaffold-module` | Create new packages or submodules |
| `/verify-plan` | Verify implementation plan before execution |
| `/using-triton-inference-server` | Triton config.pbtxt, tritonclient API, model repository |
| `/using-nvidia-dali` | DALI GPU preprocessing pipelines |
| `/using-ultralytics` | YOLO predict/export reference |
| `/uv-package-manager` | uv commands, workspace, deps, lockfile management |

### Agent team (`.claude/agents/`)

| Agent | Role | When to use |
|---|---|---|
| **architect** | System architecture, data flow, API contracts | Planning features, designing integrations, architectural decisions |
| **implementer** | Production code, strict typing, project conventions | Implementing features, fixing bugs, writing modules |
| **tester** | Tests, mocks, fixtures, coverage | Adding tests, investigating failures, test infrastructure |
| **reviewer** | Code review, security, performance, conventions | After writing code, reviewing PRs, auditing quality |

### Hooks (`.claude/hooks/`, wired in `.claude/settings.json`)

- `block-bare-tools.sh` (PreToolUse Bash) — blocks bare Python tools and `.venv/bin/`; use `uv run`.
- `block-no-verify.sh` (PreToolUse, `git commit*--no-verify*`) — blocks `--no-verify`.
- `validate-commit-msg.sh` (PreToolUse, `git commit*`) — checks the Conventional Commit subject (`-m` or `-F`).
- `pre-commit-guard.sh` (PreToolUse, `git commit*`, 300 s) — runs `uv run pre-commit run --all-files`; it does not reliably block on failure, so check its output.
- `auto-ruff-format.sh` (PostToolUse Write|Edit) — runs `ruff format` and `ruff check` on edited `.py` files.
- `inject-session-context.sh` (UserPromptSubmit, once) — injects branch, dirty-file count and `TRITON_VERSION` from `.env`.
- `setup-worktree.sh` — intended as a WorktreeCreate hook (copies `.env`, runs `uv sync --all-groups`); not registered in `settings.json`, so it never runs.
