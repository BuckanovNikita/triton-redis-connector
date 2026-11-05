---
name: commit
description: >
  Runs the triton-redis-connector commit-and-push workflow: reviews the diff, runs the pre-commit pipeline (ruff, mypy, pyright, import-linter, vulture, pytest, uv lock), updates affected docs, stages explicit task-owned paths, writes a Conventional Commit and pushes over SSH. Use when the user asks to commit, push or run /commit in this repository.
---

# Commit Workflow

## 1. Review changes

Run `git diff --stat` and `git status` to understand what changed.

## 2. Fix any failures

In Claude Code, `auto-ruff-format.sh` formats edited `.py` files. A PreToolUse hook also runs pre-commit on `git commit`, but it does not reliably block on failure, so pre-commit must pass before you commit:

```bash
uv run pre-commit run --all-files
```

- **ruff check** — try `uv run ruff check .` first, then fix manually
- **mypy** — this project uses `strict = true`; all code must be fully typed
- **pyright** — strict mode; resolve all type errors
- **vulture** — remove dead code or add whitelist entry if false positive
- **pytest** — tests run in parallel (`-n auto`); fix failing tests before committing

## 3. Update documentation

Review changes and update all project documentation (README.md and all .md at root folder) files that need to be actualized with changes.

## 4. Stage and commit

```bash
git add <paths>
git commit -m "<type>: <description>"
```

Stage explicit task-owned paths; stage everything only when every change is task-owned. For a multi-line message, write a unique temp file `/tmp/COMMIT_MSG_<short_hash>`, run `git commit -F <path>`, then delete it.

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `test:` — adding or updating tests
- `docs:` — documentation only
- `chore:` — maintenance (deps, config, CI)

The commit-message hook also accepts perf, ci, build, style and revert; see AGENTS.md.

Keep subject line concise. Use body for details when needed.

In Claude Code, hooks validate the commit message format and run pre-commit before the commit executes (pre-commit failures are not reliably blocked; check the output).

## 5. Push via SSH

```bash
git push
```

Remote must use SSH format (`git@github.com:<org>/<repo>.git`). If remote is HTTPS, fix with:

```bash
git remote set-url origin git@github.com:<org>/<repo>.git
```
