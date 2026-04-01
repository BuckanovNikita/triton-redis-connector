---
name: phase-task
description: >
  Phase-aware task navigator for the final implementation plan. Use when the
  user asks "what's next", "current phase", "implement phase N", "phase status",
  "plan progress", or "/phase-task". Tracks progress across the 9-phase final
  plan (docs/final-plan.md), identifies next unblocked work, and outputs
  structured task briefs for agents.
---

# Phase Task Navigator

Navigate the implementation plan (`docs/final-plan.md`), track deliverable progress, and identify the next unblocked work item.

## Execution Waves

The plan uses **waves** for parallelism, not sequential phases:

```
Wave 0 — ~25 tasks in parallel (no dependencies)
Wave 1 — depends on specific Wave 0 tasks
Wave 2 — depends on specific Wave 1 tasks
Wave 3 — final (E2E, docs, verification)
```

Phases 0-8 organize work by domain, waves organize execution order.

## Step 1: Read the Plan

Read `docs/final-plan.md` in full. Extract for each phase (0 through 8):

- **Phase number and name** (e.g., "Phase 2 — Export correctness")
- **Goal** (the summary under each phase heading)
- **Deliverables** (numbered subsections like 2.1, 2.2, etc.)
- **Target files** mentioned for each deliverable
- **Dependencies** — check the wave dependency graph at the bottom of the plan
- **Verification section** (commands to run when phase is done)

## Step 2: Scan Codebase for Existing Deliverables

For each deliverable from the plan, check whether the work already exists:

1. **Use Glob** to find source files mentioned in the plan under `packages/`
2. **Read key files** to check if they contain real implementation (not just placeholders)
3. **Check for tests** — deliverables with corresponding test files are more likely complete
4. **Check config files** — verify if `pyproject.toml`, `docker-compose.yml`, `pyrightconfig.json` have been updated as the plan specifies

Mark each deliverable as:
- **done** — files exist with non-placeholder implementation and tests pass
- **partial** — files exist but incomplete (missing tests, stub implementations)
- **not started** — target files do not exist or are placeholders

## Step 3: Produce Status Report

Output a markdown table:

```
| Phase | Name                            | Status      | Done | Total | Wave | Blocker      |
|-------|---------------------------------|-------------|------|-------|------|--------------|
| 0     | Cleanup & docs                  | in-progress | 2/4  | 4     | 0    | —            |
| 1     | Test audit                      | not started | 0/1  | 1     | 0    | —            |
| 2     | Export correctness              | partial     | 1/7  | 7     | 0+1  | 2.3 needs 2.2a |
| 3     | Connector improvements          | not started | 0/4  | 4     | 0+1  | —            |
| 4     | Preprocessing verification      | not started | 0/5  | 5     | 0+1  | 4.1 needs 2.2 |
| 5     | Pipeline framework              | not started | 0/7  | 7     | 0+1  | —            |
| 6     | Example simplification          | blocked     | 0/5  | 5     | 1+2  | 6.1 needs 0.1a |
| 7     | Benchmarks                      | not started | 0/6  | 6     | 0+2  | —            |
| 8     | E2E testing & final             | blocked     | 0/5  | 5     | 3    | needs 2.3, 6.4 |
```

## Step 4: Identify Next Work

1. Check the **wave dependency graph** at the end of `docs/final-plan.md`
2. Find tasks in the current wave whose dependencies are satisfied
3. If the user asked about a specific phase, focus there — but warn if blocked

## Step 5: Output Task Brief

For the identified next deliverable, output a structured brief:

```
## Task Brief

**Phase**: 2 — Export correctness
**Deliverable**: 2.1 — Fix rect bug
**Wave**: 0 (no dependencies)
**Goal**: Fix converter.py:61-62 so rect=False is passed to ultralytics

**Target files**:
- `packages/triton_model_tools/triton_model_tools/export/converter.py`

**Acceptance criteria** (from plan):
- `merged_kwargs["rect"] = rect` (always pass value)
- Add `--no-rect` to CLI

**Verification**:
```bash
uv run pytest && uv run pre-commit run --all-files
```

**Recommended agent**: implementer
```

## Agent Selection Guide

| Work Type | Agent | When |
|-----------|-------|------|
| Architecture decisions | architect | Designing new protocols, API contracts, data flow |
| Production code | implementer | Writing module implementations |
| Tests | tester | Writing unit/integration tests |
| Quality review | reviewer | After implementation, before commit |

## Notes

- Always read the CURRENT state of `docs/final-plan.md` — it may have been updated
- The plan has 26 tasks across 9 phases organized into 4 execution waves
- Key principle: **tool-first verification** — use `triton-model-tools` CLI for all verification steps
- GPU memory limit: never load multiple models simultaneously in E2E tests
- Consistency: every export/DALI/preprocessing change must be tested with both 640x640 AND 960x608
