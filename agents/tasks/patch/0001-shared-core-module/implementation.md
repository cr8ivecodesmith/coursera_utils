# Shared Core Module — Implementation (Mini)

## Understanding
- Create a `study_utils.core` package that consolidates shared helpers (OpenAI client setup, file discovery, extension parsing, UTF-8 reads) so other modules stop importing from each other.
- Accept short-term breakages: downstream commands and tests may fail until their logic is aligned in a follow-up fix; keep track of those gaps.

## Impact
- Files/modules: new `src/study_utils/core/` package (e.g., `__init__.py`, `ai.py`, `files.py`), updates to `transcribe_video`, `text_combiner`, `generate_document`, `markdown_to_pdf`, `quizzer` modules, and corresponding tests.
- Tests: add unit coverage for the core helpers; update existing discovery/AI tests; allow failing suites for downstream callers but document them.

## Plan
- [ ] Extract shared helpers into the new `core` package with module-level docs.
- [ ] Update downstream modules to consume `study_utils.core` helpers and remove duplicated code.
- [ ] Add core-focused unit tests and note any remaining failing downstream tests for the follow-up patch.

## Tests
- [ ] Unit
- [ ] Integration (if any)

## History
### 2025-09-26 17:58
**Summary**
- Drafted lightweight implementation plan covering extraction steps, consumer updates, and testing approach.
### 2025-09-26 18:13
**Summary**
- Added `study_utils.core` package, refactored downstream modules to use shared helpers, introduced core unit tests, and observed pytest teardown plugin timeouts.
