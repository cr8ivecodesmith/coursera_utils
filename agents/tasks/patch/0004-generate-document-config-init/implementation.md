# Generate Document Config Init — Implementation

## Understanding
- Extend the AI document generator by packaging it as `study_utils.generate_document`, ensuring the new package re-exports the existing public API from `__init__`, adding a `study generate-document config init` scaffold that mirrors convert-markdown, defaulting config discovery to the workspace, bundling a TOML template via `config_templates`, updating CLI wiring/docs, and culling the legacy module shim plus redundant tests.
- Public API to re-export from `generate_document/__init__.py`: `generate_document`, `find_config_path`, `load_documents_config`, `build_reference_block`, `build_messages`, `build_arg_parser`, and `main`, matching today’s `import study_utils.generate_document as gd` usage in tests and downstream scripts.
- Assumptions / Open questions: convert-markdown’s config-init ergonomics remain the desired reference; existing `documents.toml` content is sufficient for the new template without schema changes; no need to preserve the old module path for imports beyond the package namespace itself.
- Template parity check: convert-markdown ships `src/study_utils/convert_markdown/template.toml` with `[paths]`, `[execution]`, and `[logging]` tables, while generate-document relies on `documents.toml` keyed by document type. The structures intentionally differ, so the refactor will relocate the current `documents.toml` into the package template unchanged while registering it with `config_templates`.
- Risks & mitigations: CLI dispatch could break if the package boundary moves—cover with end-to-end CLI tests for both `study generate-document` and the new `config init`; packaging might omit the template—add unit coverage around `config_templates.get_template("generate_document")` and update `pyproject.toml`; config resolution regressions could surface when prioritizing the workspace—author focused tests on resolver precedence and error messaging.

## Resources
### Project docs
- agents/tasks/patch/0004-generate-document-config-init/spec.md — scope, behaviors, and constraints for the refactor.
- agents/guides/workflow.md — collaboration loop and implementation log requirements.
- agents/guides/engineering-guide.md — seam-first patterns for teasing apart CLI, config, and runner modules.
- agents/guides/patterns-and-architecture.md — guidance on package layout and CLI composition.
- agents/guides/styleguides.md — Ruff, pytest, and documentation conventions.
- agents/tasks/feat/0007-document-to-markdown/implementation.md — precedent for config-init packaging and workspace-aware commands.
### External docs
- https://docs.python.org/3/library/importlib.resources.html — ensure packaged template loading via importlib resources remains correct.

## Impact Analysis
### Affected behaviors & tests
- Config scaffolding command should write `documents.toml`, honor `--path`, `--workspace`, and `--force`, and emit the destination path—add CLI tests covering success, overwrite refusal, and forced overwrite.
- Generate-document CLI must prefer the workspace config, respect `--config`, and surface an actionable error instructing `config init` when nothing is found—update CLI integration tests and unit tests for the resolver.
- Workspace bootstrapping via `study init` should surface the new template through the shared registry—augment workspace/init tests to assert availability as needed.
- Module refactor should keep prompt-building and generation seams testable—port and update existing unit tests to the new module structure, removing obsolete shim-specific cases.
- README quickstart and command tables must reflect the new workflow—include documentation checks in review.
### Affected source files
- Create: `src/study_utils/generate_document/__init__.py`, `src/study_utils/generate_document/cli.py`, `src/study_utils/generate_document/config.py`, `src/study_utils/generate_document/template.toml`, potentially `src/study_utils/generate_document/runner.py` (or similar) for the core generation orchestration, with `__init__` re-exporting the consumer-facing API the legacy module provided.
- Modify: `src/study_utils/cli.py` to point at the new package entry point; `src/study_utils/core/config_templates.py` to register the generate-document template; `pyproject.toml` package-data stanza; README quickstart/docs; tests under `tests/` to target the new modules; any shared helpers relying on the old module path.
- Delete: `src/study_utils/generate_document.py` legacy shim along with redundant alias tests/fixtures no longer meaningful after the package split.
- Config/flags: ensure new CLI flags mirror convert-markdown (`--path`, `--workspace`, `--force`) without introducing additional toggles.
### Security considerations
- Maintain local-only file I/O and 0o600 permissions when writing configs; continue to avoid logging prompt contents or API keys; ensure new error paths do not leak sensitive data.

## Solution Plan
- Architecture/pattern choices: follow convert-markdown’s package layout to keep CLI, config resolution, and execution seams isolated, aligning with the seam-first guidance in `patterns-and-architecture.md`; leverage `importlib.resources` for template access and shared workspace helpers for default paths, reusing the shared `config_templates` scaffolding helpers to avoid duplicating file-writing logic.
- DI & boundaries: expose dependency seams for OpenAI client loading and filesystem interactions via injected callables (as today) so unit tests can stub them; keep CLI parsing separate from the generation runner per `engineering-guide.md` to simplify coverage and reuse.
- Stepwise checklist:
  - [x] Introduce the `study_utils.generate_document` package structure, migrating core helpers into `config.py` and `runner.py` while keeping pure functions intact.
  - [x] Wire `cli.py` to provide both the document generation command and a nested `config init` subcommand mirroring convert-markdown options, keeping existing flags intact.
  - [x] Register the bundled `template.toml` in `core.config_templates` and move existing TOML content into the new resource via the shared scaffolding helper.
  - [x] Ensure `generate_document/__init__.py` re-exports the consumer-facing API the legacy module provided so imports stay stable without the shim.
  - [ ] Update `study_utils.cli` dispatch, remove the old module file, and adjust any imports/tests that expect the previous shim.
  - [ ] Refresh README quickstart and command documentation to highlight `study generate-document config init` and updated config resolution behavior.
  - [ ] Expand/adjust tests: unit tests for config resolution precedence and error messaging, CLI tests for the new command group (covering `--config`, `--workspace`, and `--force`), and cleanup of obsolete shim fixtures.
  - [ ] Update packaging metadata so the new template ships with the wheel and eliminate the old `documents.toml` entry.
  - [ ] Run `uv run pytest` (or equivalent) to maintain 100% coverage once refactor is complete.
  - [ ] Run `uv run ruff check` to ensure lint parity after module moves.

## Test Plan
- Unit: cover config resolver precedence (workspace > cwd > bundled), template writer behavior (`--force` and existing files), runner validation errors for missing doc types or empty completions, and pure helpers like message building after the move.
- Integration/E2E: CLI invocations for `study generate-document` happy path with stubbed OpenAI client, error path when config missing suggesting `config init`, and full CLI coverage for `study generate-document config init` options.
- Manual checks: invoke `study generate-document config init` against a temp workspace and run `study generate-document` end-to-end with sample inputs to ensure messaging and file outputs look correct.

## Operability
- Ensure the CLI reports destination paths and actionable guidance consistently; update README quickstart plus any other user-facing docs; verify `study list` output remains accurate; document revert steps by noting removal of new package modules and template registrations.

## History
### 2025-09-30 14:15
**Summary** — Initial implementation plan for generate-document config init refactor
**Changes**
- Captured understanding, solution outline, test coverage, and operational considerations for packaging generate-document with a config-init workflow.

### 2025-10-01 20:07
**Summary** — Established generate-document package scaffolding and API re-exports
**Changes**
- Created `study_utils/generate_document` package (`cli.py`, `config.py`, `runner.py`, `__init__.py`) mirroring the legacy module helpers and public surface.
- Relocated `documents.toml` alongside the new package and updated tests/imports to target the package modules.

### 2025-10-02 11:20
**Summary** — Added CLI config command scaffolding for generate-document
**Changes**
- Extended `generate_document.cli.main` to route `study generate-document config init` with `--path`, `--workspace`, and `--force` support mirroring convert-markdown.
- Plumbed `config_templates`/workspace helpers and introduced `CONFIG_FILENAME` constant for shared defaults pending template registration.

### 2025-10-02 15:45
**Summary** — Registered generate-document template with shared registry
**Changes**
- Added `study_utils.generate_document` entry to `core.config_templates` so config scaffolding uses the shared helper.
- Kept existing `documents.toml` content as the packaged template; no tests run yet (full suite pending later checklist steps).

### 2025-10-02 17:05
**Summary** — Locked generate-document package exports to legacy surface
**Changes**
- Trimmed `study_utils.generate_document.__all__` to the historically supported API (`generate_document`, `find_config_path`, `load_documents_config`, `build_reference_block`, `build_messages`, `build_arg_parser`, `main`).
- Ensured downstream imports via `import study_utils.generate_document as gd` remain stable ahead of removing the legacy shim.
