---
name: triton-workspace-dependencies
description: Triton workspace dependency and mutually exclusive version-extra procedures.
---

# Triton workspace dependencies

Use `uv run` for project tools. The canonical verification entrypoint is
`uv run prek run --all-files`; its configuration owns the tool inventory.

Workspace members under `packages/` use workspace sources. Keep development dependencies
in dependency groups and commit `uv.lock` with dependency changes. Preserve the separate
lint and test groups included by dev.

Triton version extras are mutually exclusive: choose the extra matching `TRITON_VERSION`
and follow `docs/triton-versions.md` when switching. Do not use `--all-extras`.
Use the global uv-package-manager skill for general uv behavior; this file owns only
Triton-specific constraints.
