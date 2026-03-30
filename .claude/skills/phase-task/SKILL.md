---
name: phase-task
description: >
  Phase-aware task navigator for the unified implementation plan. Use when the
  user asks "what's next", "current phase", "implement phase N", "phase status",
  "plan progress", or "/phase-task". Tracks progress across the 7-phase unified
  plan (docs/unified-plan.md), identifies next unblocked work, and outputs
  structured task briefs for agents.
---

# Phase Task Navigator

Navigate the unified implementation plan (`docs/unified-plan.md`), track deliverable progress, and identify the next unblocked work item.

## Phase Dependency Graph

```
Phase 0 (Infrastructure)
  └─► Phase 1 (Foundation)
        ├─► Phase 2  (Export + TensorRT + DALI)
        ├─► Phase 3A (App connectors, engine, main loop)
        └─► Phase 3B (Tools inference client)
              └─► Phase 4 (Inference worker pool + E2E)
                    └─► Phase 5 (Benchmarking)
                          └─► Phase 6 (Metrics + Comparison)
                                └─► Phase 7 (Aggregation + Polish)
```

Phases 2, 3A, and 3B can run in parallel after Phase 1 is complete.

## Step 1: Read the Unified Plan

Read `docs/unified-plan.md` in full. Extract for each phase (0 through 7):

- **Phase number and name** (e.g., "Phase 1 — Foundation")
- **Goal** (the one-line summary under each phase heading)
- **Deliverables** (numbered subsections like 1.1, 1.2, etc.)
- **Target files** mentioned for each deliverable
- **Dependencies** on prior phases
- **Verification section** (commands to run when phase is done)

## Step 2: Scan Codebase for Existing Deliverables

For each deliverable from the plan, check whether the work already exists:

1. **Use Glob** to find source files mentioned in the plan under `packages/`
2. **Read key files** to check if they contain real implementation (not just placeholders like `"""Placeholder."""`)
3. **Check for tests** — deliverables with corresponding test files in `tests/` are more likely complete
4. **Check config files** — verify if `pyproject.toml`, `docker-compose.yml`, `pyrightconfig.json` have been updated as the plan specifies

Mark each deliverable as:
- **done** — files exist with non-placeholder implementation and tests pass
- **partial** — files exist but incomplete (missing tests, stub implementations)
- **not started** — target files do not exist or are placeholders

## Step 3: Produce Status Report

Output a markdown table:

```
| Phase | Name                           | Status      | Done | Total | Blocker          |
|-------|--------------------------------|-------------|------|-------|------------------|
| 0     | Infrastructure & Versions      | complete    | 3/3  | 3     | —                |
| 1     | Foundation                     | in-progress | 2/8  | 8     | —                |
| 2     | Export + TensorRT + DALI       | blocked     | 0/5  | 5     | Phase 1          |
| 3A    | App Connectors + Engine        | blocked     | 0/6  | 6     | Phase 1          |
| 3B    | Tools Inference Client         | blocked     | 0/4  | 4     | Phase 1          |
| 4     | Inference Pool + E2E           | blocked     | 0/3  | 3     | Phases 3A, 3B    |
| 5     | Benchmarking                   | blocked     | 0/3  | 3     | Phase 4          |
| 6     | Metrics + Comparison           | blocked     | 0/4  | 4     | Phase 5          |
| 7     | Aggregation + Polish           | blocked     | 0/3  | 3     | Phase 6          |
```

**Status rules**:
- **complete** — all deliverables done, verification commands pass
- **in-progress** — at least one deliverable done, phase is unblocked
- **not started** — no deliverables done, phase is unblocked
- **blocked** — prerequisite phases are not complete

## Step 4: Identify Next Work

1. Find the first phase that is **not complete** and **not blocked**
2. Within that phase, find the first deliverable that is **not done**
3. If the user asked about a specific phase (e.g., "phase 3A"), focus on that phase regardless — but warn if it is blocked

## Step 5: Output Task Brief

For the identified next deliverable (or user-requested phase), output a structured brief:

```
## Task Brief

**Phase**: 1 — Foundation
**Deliverable**: 1.3 — Envelope event wrapper
**Goal**: Create the Envelope[T] generic wrapper for typed event routing

**Target files**:
- `packages/triton_redis_connector/triton_redis_connector/envelope.py`
- `packages/triton_redis_connector/tests/test_envelope.py`

**Dependencies**: Deliverables 1.1 (shared models package) must exist

**Acceptance criteria** (from plan):
- Generic Envelope[T] with metadata fields
- Timestamp tracking for transit metrics
- Type-safe wrapping/unwrapping

**Tests required**:
- Unit tests for Envelope creation, wrapping, unwrapping
- Type narrowing tests

**Verification**:
```bash
uv run pytest && uv run pre-commit run --all-files
```

**Recommended agent**: implementer (production code task)
```

## Agent Selection Guide

| Work Type | Agent | When |
|-----------|-------|------|
| Protocol/interface design | architect | Defining new protocols, API contracts, data flow |
| Production code | implementer | Writing module implementations |
| Tests | tester | Writing unit/integration tests |
| Quality review | reviewer | After implementation, before commit |

## Notes

- Always read the CURRENT state of `docs/unified-plan.md` — it may have been updated since the last check
- The plan has ~821 lines across 7 phases with internal sub-deliverables
- Phase 4C (shared memory IPC) is explicitly deferred until after Phase 5 benchmarks confirm it is needed
- The LetterBox preprocessing correctness constraint spans Phases 2, 3A, and 3B — flag this cross-cutting concern when any of these phases are active
