# Study CLI Wrapper — Implementation Plan

## Understanding
- Build a unified `study` CLI that routes to existing tool entrypoints and removes the standalone scripts.
- Ensure the new wrapper exposes listing/help/version utilities and covers the missing `markdown-to-pdf` entry point.
- Confirm no aliases are introduced and `study help` mirrors the default usage banner.

## Risks & Assumptions
- Removing legacy console scripts will break existing automation until updated; plan test updates alongside code.
- Need to avoid importing heavy modules on startup; prefer lightweight registration.
- Quizzer TUI must not launch during tests—requires mocking or guard rails.

## Impact
- New module `src/study_utils/cli.py` plus tests under `tests/`.
- Update `pyproject.toml` scripts block; adjust README CLI docs.
- Remove or refactor any tests targeting the old console entry points.

## Milestones

- [x] **CLI Skeleton & Command Metadata**  
  - Add `study_utils.cli` with command registry, usage/list/help stubs, and version plumbing.  
  - Introduce unit tests for usage rendering, `list`, `help`, and `--version` using mocks where needed.  
  - *Tests*: new `tests/test_cli.py` covering the above behaviors.

- [x] **Subcommand Dispatch Wiring**  
  - Implement dispatch logic that forwards argv to each tool’s `main`, handling exit codes and `--` passthrough.  
  - Add tests that patch the tool `main` functions to assert call semantics and error handling for unknown commands.  
  - *Tests*: extend CLI test suite to cover dispatch paths.

- [ ] **Packaging Updates**  
  - Replace `[project.scripts]` entries with `study` and `markdown-to-pdf`; drop legacy script names.  
  - Remove or adapt existing tests referencing old console scripts.  
  - *Tests*: run CLI unit suite and any packaging-related checks (e.g., `uv run pytest`).

- [ ] **Documentation Refresh**  
  - Update README CLI section with `study` usage examples and note the consolidation.  
  - Ensure no references to removed scripts remain.  
  - *Tests*: spell-check/docs lint if available (manual review otherwise).

## Tests
- [ ] CLI unit tests (`pytest`) validating new behavior.
- [ ] Integration/regression tests (N/A unless a higher-level CLI invocation test is added).

## History
### 2025-02-14 hh:mm
**Summary**
- Created initial CLI skeleton with usage/list/help/version handling and added unit coverage in `tests/test_cli.py`.

### 2025-09-26 15:37 UTC
**Summary**
- Wired `study` subcommand dispatch to forward args/exit codes to tool `main` functions and enforce passthrough semantics.
- Expanded `tests/test_cli.py` with handler stubs covering success, SystemExit variants, and edge cases so coverage stays at 100%.
