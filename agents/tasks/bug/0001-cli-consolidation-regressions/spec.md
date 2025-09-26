# CLI consolidation regressions — Spec (Mini)

## Description
Audit `study` subcommands for signature or option mismatches left over from pre-consolidation wrappers, fixing runtime errors like the `parse_extensions(... default=...)` TypeError.

## Behavior
- Given the consolidated `study` CLI with bundled subcommands
- When I invoke each documented subcommand using its advertised flags (e.g. `study generate-document ...`, `study text-combiner ...`)
- Then the command runs without TypeErrors or similar runtime failures caused by stale wrappers or incompatible argument defaults

## Notes
- Review every subcommand wrapper that re-exports helpers from `study_utils.core`; ensure signatures and defaults match their callers after consolidation.
- Add integration-style CLI tests that exercise representative invocations for each subcommand to catch future signature drift.
- Confirm docs/help output still reflects the supported arguments once fixes land.
- Backward compatibility with standalone scripts is not required; prune obsolete code paths and tests that only served legacy entrypoints.

## History
### 2025-09-27 04:13
**Summary**
- Captured regression after CLI consolidation where wrapper signatures diverged from internal helpers, causing runtime errors in real usage.
