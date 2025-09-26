# Study Utils Test Coverage Lift — Implementation

## Understanding
- Raise overall `study_utils` coverage to 100% by backfilling tests, adding seams around external dependencies (OpenAI, WeasyPrint, Textual, Pydub), and enforcing a permanent coverage gate. Maintain offline, deterministic test runs.
- Assumptions / Open questions
  - Existing CI can surface coverage failures once `--cov-fail-under=100` is enabled; confirm pipeline command alignment before final milestone.
  - Some modules may need light refactors (e.g., dependency injection) for testability; scope to minimal surface changes.
  - Textual UI testing will rely on headless harnessing or pure helper coverage; no interactive UI tests planned.
- Risks & mitigations
  - Heavy dependency mocks could drift from real APIs → centralize fixtures with comments mirroring real responses; keep them low-level and reusable.
  - Introducing seams might break runtime behavior → add regression unit tests and, where feasible, smoke CLI tests for success paths.
  - Coverage gate could block unrelated work if brittle → ensure flaky areas (e.g., random seeds) are seeded and deterministic before enabling gate.

## Resources
### Project docs
- agents/guides/engineering-guide.md — seam-first, DI, and testing defaults.
- agents/guides/patterns-and-architecture.md — guidance on module organization while adding seams/mocks.
- agents/guides/styleguides.md — pytest conventions, docstring patterns, Ruff alignment.
- agents/guides/workflow.md — collaboration loop, history updates, milestone commits.
### External docs
- https://docs.pytest.org/en/stable/how-to/parametrize.html — patterns for broadening coverage via parametrization.
- https://coverage.readthedocs.io/en/latest/config.html — configuring coverage thresholds and reports.
- https://textual.textualize.io/guide/testing/ — strategies for testing Textual apps without UI interaction.

## Impact Analysis
### Affected behaviors & tests
- Core utilities (extensions parsing, file iterators) → expand unit suites (`tests/test_core_files.py`, add new tests).
- AI client loaders and dependent modules → add tests with patched OpenAI clients across `core`, `generate_document`, `text_combiner`, `quizzer` manager/utils, `transcribe_video`.
- CLI entrypoints (generate_document, markdown_to_pdf, quizzer, text_combiner, transcription) → add `SystemExit`-based unit tests covering success and error flows.
- Markdown/PDF pipeline → ensure every helper (slugify, TOC, CSS builder) has direct tests.
- Quizzer view/helpers → cover aggregation and navigation logic without real UI.
### Affected source files
- Create: `tests/fixtures/` helpers (e.g., `tests/fixtures/openai.py`, `tests/fixtures/weasyprint.py`), potential `tests/utils.py` for shared factories.
- Modify: targeted modules for seam injection (likely `core/ai.py`, `markdown_to_pdf.py`, `quizzer/utils.py`, `text_combiner.py`, `transcribe_video.py`, CLI modules) and multiple test files.
- Config/flags: update `pyproject.toml` or `pytest.ini` for coverage threshold; adjust `justfile` test target as needed.
### Security considerations
- Ensure mocks never touch real API keys; guard with environment-variable assertions in tests.
- Temporary files use pytest tmp_path to avoid leaking artifacts.

## Solution Plan
- Architecture/pattern choices: leverage seam-first approach (engineering guide) by injecting clients via parameters or helper functions; prefer pure helpers for complex logic (patterns guide).
- DI & boundaries: introduce optional parameters or helper factories for external clients (OpenAI, WeasyPrint) to replace hard-coded imports during testing.
- Stepwise checklist (milestone commit per bullet)
  - [x] **Milestone 1 – Testing toolkit baseline**: add shared fixtures/mocks (OpenAI stub, WeasyPrint stub, tmp workspace helpers), ensure `pytest --cov` wiring captured in `justfile` and `pyproject` coverage config; run suite (expect still <100%).
  - [ ] **Milestone 2 – Core coverage**: backfill tests for `core/ai.py`, `core/files.py`, and other foundational helpers; adjust code for seamability as required; commit once coverage for these modules hits 100%.
  - [ ] **Milestone 3 – Document generation & markdown pipeline**: expand tests across `generate_document.py` and `markdown_to_pdf.py` helpers/CLI; stub WeasyPrint interactions; verify partial coverage jump; commit.
  - [ ] **Milestone 4 – Quizzer domain**: cover `quizzer/utils.py`, CLI command handlers, manager (question generation, topic extraction), and view helpers with deterministic seeds/mocks; add minimal CLI smoke tests; commit.
  - [ ] **Milestone 5 – Text utilities & transcription**: add tests for `text_combiner.py` and `transcribe_video.py` including smart naming, cache IO, CLI flows, and AI fallbacks; ensure deterministic chunking via patched pydub; commit.
  - [ ] **Milestone 6 – Final hardening & gate**: sweep remaining uncovered lines (CLI `__main__` modules, residual branches), enable `--cov-fail-under=100`, add documentation updates, confirm CI alignment, and perform final commit.

## Test Plan
- Unit: cover every helper/branch listed above using parametrized tests and patched dependencies.
- Contract: validate stub client interfaces mimic required `chat.completions.create`/WeasyPrint methods so production code remains compatible.
- Integration/E2E: CLI invocation tests using `capsys` + `pytest.raises(SystemExit)` covering success/error; ensure they rely on tmp directories.
- Manual checks: Run `pytest --cov=study_utils --cov-report=term-missing` after each milestone; optionally run `coverage html` locally to inspect stragglers.

## Operability
- Telemetry: none added; rely on coverage gate.
- Runbooks / revert: document in README/testing section how to disable coverage gate temporarily if urgently required (manual edit of `pyproject`), but expect no runtime impact.

## History
### 2024-05-23 00:00
**Summary**
Initial implementation plan drafted.
**Changes**
- Captured milestone-based checklist aligned with spec goals.

### 2025-09-26 11:52
**Summary**
Milestone 1 testing toolkit baseline complete.
**Changes**
- Added shared stub modules (openai, weasyprint, dotenv, pydub) and workspace helper.
- Updated pytest coverage reporting to emit coverage.xml.
- Documented coverage artifact in .gitignore.
