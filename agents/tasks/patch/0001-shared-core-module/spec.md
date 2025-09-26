# Shared Core Module — Spec

## Summary
Refactor `study_utils` to introduce a dedicated `core` package that centralizes shared helpers (OpenAI client setup, text-file discovery, extension parsing, and UTF-8 file I/O) so callers no longer reach into sibling modules for cross-cutting behavior.

## Goals
- Create `src/study_utils/core/` with focused modules (e.g., `ai.py`, `files.py`, optional `text.py`) and an `__init__` exporting stable helpers.
- Move the existing `load_client` helper and its dotenv handling into `core.ai` so any tool can import it without the `transcribe_video` dependency.
- Consolidate duplicated file-discovery, ordering, extension-normalization, and text-reading helpers used by `text_combiner`, `generate_document`, `markdown_to_pdf`, and `quizzer` into `core.files` (or similar) with consistent error handling.
- Update affected modules to import from the new core package, even if intermediate behavioral gaps surface in the short term.
- Add targeted unit coverage for the new core helpers while documenting downstream tests expected to fail until their modules are adapted in a follow-up patch.

## Non-Goals
- No redesign of individual CLI interfaces or option semantics beyond import source changes.
- No consolidation of domain-specific logic (e.g., quiz generation, markdown rendering) outside the targeted shared helpers.
- No introduction of new external dependencies or configuration formats.

## Behavior (BDD-ish)
- Given any module needs an OpenAI client, when it imports and calls `study_utils.core.ai.load_client()`, then env loading, error messaging, and client construction stay functionally equivalent even if CLI text shifts slightly.
- Given downstream commands normalize extension arguments, when they call the new shared parser with custom defaults, then return values may shift until their callers are adjusted, and those regressions should be captured by failing tests.
- Given discovery helpers receive files and directories with depth limits, when invoked via the new core API, then traversal logic aligns with the new shared implementation and exposes gaps in downstream expectations for later repair.

## Constraints & Dependencies
- Constraints: minimize unnecessary coupling and avoid importing heavy optional dependencies (WeasyPrint, OpenAI) unless lazily needed; backward compatibility is not required for this patch.
- Upstream/Downstream: track `transcribe_video`, `generate_document`, `text_combiner`, `markdown_to_pdf`, and `quizzer` for breakages; allow tests to fail temporarily and catalog them for the follow-up fix.

## Security & Privacy
- Continue to source secrets via environment variables and `.env`; avoid logging API keys; ensure the shared helper raises clean errors when keys are missing.

## Telemetry & Operability
- No new telemetry required; reuse existing logging or stdout messages. Document any new debug logs if added.

## Rollout / Revert
- Rollout: add core package, migrate imports module-by-module, record failing tests/behaviors that require remediation in the next patch.
- Revert: restore prior helper definitions in their original modules and delete the new core package.

## Definition of Done
- [ ] Documented list of failing downstream tests/behaviors caused by the migration
- [ ] Docs updated where modules previously referenced `transcribe_video.load_client`
- [ ] Minimal unit coverage for new core helpers in place
- [ ] Flags defaulted per channel (n/a; ensure left unchecked unless needed)
- [ ] Monitoring in place (n/a; leave unchecked)

## Ownership
- Owner: @matt
- Reviewers: @codex
- Stakeholders: @matt

## Links
- Related code: `src/study_utils/transcribe_video.py`, `src/study_utils/text_combiner.py`, `src/study_utils/generate_document.py`, `src/study_utils/markdown_to_pdf.py`, `src/study_utils/quizzer/utils.py`, `tests/test_text_combiner.py`, `tests/test_generate_document.py`, `tests/test_quizzer.py`

## History
### 2025-09-26 17:47
**Summary** — Initial spec drafted for shared core module refactor
**Changes**
- Captured goals, scope, and rollout plan for consolidating shared helpers into `study_utils.core`
### 2025-09-26 17:54
**Summary** — Updated scope to allow intentional breakages during migration
**Changes**
- Relaxed backward-compatibility language, clarified expectations for failing tests, and adjusted Definition of Done
