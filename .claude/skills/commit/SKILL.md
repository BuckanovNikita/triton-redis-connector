---
name: commit
description: >
  Use when the user wants to commit and push code changes. Handles the full
  commit workflow: running quality checks (ruff, mypy, pyright, vulture,
  pytest via pre-commit), staging files, creating a conventional commit, and
  pushing via SSH. Trigger on: "commit", "push", "/commit".
---

# Commit Workflow

## 1. Review changes

Run `git diff --stat` and `git status` to understand what changed.

## 2. Fix any failures

Ruff format/fix runs automatically on every file save (hook).
Pre-commit runs automatically before every commit (hook).

If you want to check early, run manually:

```bash
uv run pre-commit run --all-files
```

- **ruff check** — try `uv run ruff check --fix .` first, then fix manually
- **mypy** — this project uses `strict = true`; all code must be fully typed
- **pyright** — strict mode; resolve all type errors
- **vulture** — remove dead code or add whitelist entry if false positive
- **pytest** — tests run in parallel (`-n auto`); fix failing tests before committing

## 3. Update documentation

Review changes and update all project documentation (README.md and all .md at root folder) files that need to be actualized with changes.

## 4. Stage and commit

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

Hooks auto-validate commit message format and run pre-commit before the commit executes.

## 5. Push via SSH

```bash
git push
```

Remote must use SSH format (`git@github.com:<org>/<repo>.git`). If remote is HTTPS, fix with:

```bash
git remote set-url origin git@github.com:<org>/<repo>.git
```
