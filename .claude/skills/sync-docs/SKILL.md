---
name: sync-docs
description: >
  Documentation synchronization checker. Use when updating docs after code
  changes, checking for documentation drift, marking phases as done in the
  unified plan, or "/sync-docs". Compares docs/final-plan.md, README.md,
  and CLAUDE.md against actual codebase state and produces a discrepancy report.
---

# Documentation Synchronization

Compare documentation against actual codebase state, identify discrepancies,
and update docs to match reality.

## Documents to Check

| File | Language | Content |
|------|----------|---------|
| `docs/final-plan.md` | Russian | 9-phase implementation plan (26 tasks, 4 execution waves), deliverables |
| `README.md` | Russian | Quick start, commands, infrastructure |
| `CLAUDE.md` | English | Agent guidance, workspace structure, commands |

**Note**: `docs/design.md` was removed as obsolete. A new `docs/architecture.md` will be created in Phase 8.4 after all implementation is complete.

## Step 1: Read All Documents

Read all three documentation files in full. Note the key factual claims:
- Package structure and file paths
- Dependency versions
- Infrastructure configuration (ports, images, commands)
- Phase deliverables and their status
- Development commands

## Step 2: Scan Codebase for Discrepancies

### Package structure
- Compare package structure claims in docs against actual `packages/` contents
- Use Glob to check which packages and modules actually exist
- Flag any packages/modules mentioned in docs that don't exist yet (expected for future phases)
- Flag any packages/modules that exist but aren't documented

### Dependency versions
- Compare versions in docs against `pyproject.toml` files (root and per-package)
- Check: `tritonclient`, `ultralytics`, `onnx`, `onnxruntime`, `redis`, `pydantic`, `loguru`

### Infrastructure
- Compare Docker service definitions in docs against `.devcontainer/docker-compose.yml`
- Check: image tags, ports, volume mounts, health checks, command flags

### Phase deliverables
- For each phase in `docs/final-plan.md`, check if deliverable files exist
- Mark deliverables as: exists (with real code), exists (placeholder only), missing

### Development commands
- Compare commands in `README.md` and `CLAUDE.md` against actual tool configuration
- Check pre-commit hook list matches `.pre-commit-config.yaml`

## Step 3: Produce Discrepancy Report

Output a structured report:

```markdown
## Documentation Discrepancies

### docs/final-plan.md
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
- `README.md` — Russian
- All other docs (`CLAUDE.md`, `docs/*.md`, agents, skills) — English

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
