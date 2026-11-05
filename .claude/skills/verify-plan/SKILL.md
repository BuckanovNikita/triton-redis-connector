---
name: verify-plan
description: >
  Audits implementation plans under docs/plans/ against requirements docs and the current codebase and reports gaps, contradictions and stale references in a table, without editing files. Use when asked to verify, review or sanity-check a plan before execution, or when running /verify-plan.
---
# Verify Plan
1. Read all plan files in `docs/plans/` (task directories; stop and report if none exist)
2. Cross-reference against requirements docs and current codebase
3. Identify gaps, contradictions, and stale references
4. Report findings in a structured table
5. Do NOT make changes unless explicitly asked
