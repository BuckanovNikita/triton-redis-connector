# Step 01: Update Instructions

**Agent:** implementer
**Depends on:** nothing (run first)

## Goal

Update CLAUDE.md and reviewer agent with rules to prevent the antipatterns found in new_round.md from recurring.

## Tasks

### 1. Update project CLAUDE.md

**File:** `/home/nkt/triton-redis-connector/CLAUDE.md`

Add to **Testing Policy** section:
- "Never test pydantic/framework validation (field constraints, enum values, frozen models) — these are library guarantees"
- "Never test download/network tools — one-time use utilities, not worth unit testing"
- "Never test simple file I/O (collect paths, save txt) — cover by e2e tests only"
- "E2E means real infrastructure (Triton, Redis), not mocks of infrastructure"
- "Coverage target: 70% minimum. Exclude simple utility modules from coverage."

Add to **Code Style** section:
- "Never write custom IoU or NMS — use torch ecosystem: `torchvision.ops.box_iou` / `torchvision.ops.nms` (torchvision is allowed)"
- "No re-export modules — import from source package directly, never create proxy modules that only re-export"
- "Processors/decoders must explicitly inherit protocol classes, not rely on structural typing"

Add to **Architecture** section:
- "perf_analyzer orchestration via bash scripts in `scripts/`, not Python wrappers"
- "No memory-limiting code in versioned repo — local PC restrictions belong in local scripts"
- "No `model_repository_offline` — use explicit model control mode with single `model_repository`"
- "Tool output defaults to `e2e_results/` (gitignored), never `/tmp`"
- "Dynamic batching enabled by default on model export, with `preferred_batch_size` auto-set"
- "Use tqdm for progress bars in iteration loops (benchmarks, inference, image processing)"

### 2. Update reviewer agent

**File:** `/home/nkt/triton-redis-connector/.claude/agents/reviewer.md`

Add antipattern checklist:
- Flag tests that only verify pydantic validation (field constraints, enum values)
- Flag re-export modules (files that only import and re-export from another package)
- Flag unfinished wiring (docstring says "wrapped with X" but code doesn't wrap)
- Flag custom mathematical implementations when torch/torchvision equivalents exist
- Flag `Any` type annotations that could be narrowed to proper types
- Flag lazy imports (`PLC0415`) outside of CLI subcommand files

## Verification

```bash
uv run pre-commit run --all-files
```
