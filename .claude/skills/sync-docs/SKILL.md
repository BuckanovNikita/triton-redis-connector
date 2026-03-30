---
name: sync-docs
description: >
  Documentation synchronization checker. Use when updating docs after code
  changes, checking for documentation drift, marking phases as done in the
  unified plan, or "/sync-docs". Compares docs/design.md, docs/unified-plan.md,
  README.md, and CLAUDE.md against actual codebase state and produces a
  discrepancy report.
---

# Documentation Synchronization

Compare documentation against actual codebase state, identify discrepancies,
and update docs to match reality.

## Documents to Check

| File | Language | Content |
|------|----------|---------|
| `docs/design.md` | Russian | Architecture spec, protocols, data flow |
| `docs/unified-plan.md` | Russian | 7-phase implementation plan, deliverables |
| `README.md` | Russian | Quick start, commands, infrastructure |
| `CLAUDE.md` | English | Agent guidance, workspace structure, commands |

## Step 1: Read All Documents

Read all four documentation files in full. Note the key factual claims:
- Package structure and file paths
- Dependency versions
- Infrastructure configuration (ports, images, commands)
- Protocol definitions and signatures
- Phase deliverables and their status
- Development commands

## Step 2: Scan Codebase for Discrepancies

### Package structure
- Compare `docs/design.md` package structure tree against actual `packages/` contents
- Use Glob to check which packages and modules actually exist
- Flag any packages/modules mentioned in docs that don't exist yet (expected for future phases)
- Flag any packages/modules that exist but aren't documented

### Protocol definitions
- Compare protocol signatures in docs against actual source code (if protocols are implemented)
- Check `__call__` signatures, `stop` field presence, return types

### Dependency versions
- Compare versions in docs against `pyproject.toml` files (root and per-package)
- Check: `tritonclient`, `ultralytics`, `onnx`, `onnxruntime`, `redis`, `pydantic`, `loguru`

### Infrastructure
- Compare Docker service definitions in docs against `.devcontainer/docker-compose.yml`
- Check: image tags, ports, volume mounts, health checks, command flags

### Phase deliverables
- For each phase in `docs/unified-plan.md`, check if deliverable files exist
- Mark deliverables as: exists (with real code), exists (placeholder only), missing

### Development commands
- Compare commands in `README.md` and `CLAUDE.md` against actual tool configuration
- Check pre-commit hook list matches `.pre-commit-config.yaml`

## Step 3: Produce Discrepancy Report

Output a structured report:

```markdown
## Documentation Discrepancies

### docs/design.md
- [ ] Section 2: Package structure shows `triton_shared_models` but package doesn't exist yet (Phase 1 deliverable — expected)
- [x] Section 5: `tritonclient>=2.52.0` but pyproject.toml has `>=2.66.0` — **needs update**

### docs/unified-plan.md
- [x] Phase 0: All deliverables complete — **mark as done**
- [ ] Phase 1.3: Envelope module exists but has only placeholder code

### README.md
- [x] Commands section missing `uv run lint-imports` — **needs update**

### CLAUDE.md
- [x] Pre-commit pipeline description missing import-linter step — **needs update**
```

Use `[x]` for actionable discrepancies (docs need updating) and `[ ]` for
expected gaps (future work not yet implemented).

## Step 4: Apply Updates (on user confirmation)

After the user reviews the report and confirms, update each document:

### Language rules
- `docs/design.md` — Russian
- `docs/unified-plan.md` — Russian
- `README.md` — Russian
- `CLAUDE.md` — English (always)

### Update guidelines
- Only update factual claims that are verifiably wrong (versions, file paths, commands)
- Do NOT rewrite architectural descriptions unless the architecture has actually changed
- Mark completed phase deliverables but do NOT remove future deliverable descriptions
- Preserve existing formatting and section structure
- When updating Russian docs, maintain the existing Russian language style

## Step 5: Verify

After updates, run:
```bash
uv run ruff check .  # catches any broken markdown code blocks
git diff --stat      # review what changed
```
