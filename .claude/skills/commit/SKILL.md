---
name: commit
description: >
  Use when the user wants to commit and push code changes. Handles the full
  commit workflow: running quality checks (ruff, mypy, pyright, vulture,
  pytest via pre-commit), staging files, creating a conventional commit, and
  pushing via SSH. Trigger on: "commit", "push", "/commit".
---

# Commit Workflow

## 1. Run ruff formatter and fix autofixable errors

```bash
uv run ruff format .
uv run ruff check --fix .
```

## 2. Review changes

Run `git diff --stat` and `git status` to understand what changed.

## 3. Run all quality checks

```bash
uv run pre-commit run --all-files
```

This runs in order: ruff format, ruff check, mypy, pyright, vulture, pytest, uv lock.

If pre-commit fails because of unstaged changes, stash first:

```bash
git stash
uv run pre-commit run --all-files
git stash pop
```

If ruff format modified files, re-add them and re-run.

## 4. Fix any failures

- **ruff check** — try `uv run ruff check --fix .` first, then fix manually
- **mypy** — this project uses `strict = true`; all code must be fully typed
- **pyright** — strict mode; resolve all type errors
- **vulture** — remove dead code or add whitelist entry if false positive
- **pytest** — tests run in parallel (`-n auto`); fix failing tests before committing

Re-run `uv run pre-commit run --all-files` until all hooks pass.

## 5. Update documentation

Review changes and update all project documentation (README.md and all .md at root folder) files that need to be actualized with changes.

## 6. Stage and commit

```bash
git add -A
git commit -m "<type>: <description>"
```

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `test:` — adding or updating tests
- `docs:` — documentation only
- `chore:` — maintenance (deps, config, CI)

Keep subject line concise. Use body for details when needed.

## 7. Push via SSH

```bash
git push
```

Remote must use SSH format (`git@github.com:<org>/<repo>.git`). If remote is HTTPS, fix with:

```bash
git remote set-url origin git@github.com:<org>/<repo>.git
```

## Important: use `--no-verify` only as a last resort

Always prefer fixing the issue over skipping hooks.
