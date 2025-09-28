# Document to Markdown Converter — Implementation

## Understanding
- Build a `study convert-markdown` workflow that turns TXT, PDF, DOCX, and HTML files into Markdown via Microsoft’s `markitdown`, falls back to `unstructured` for EPUB, and writes outputs (with YAML front matter) into a workspace directory by default while keeping original basenames. Configuration (extensions, output dir, overwrite/version policy, logging) lives in `convert_markdown.toml` and is resolved through a shared loader (CLI flags > environment variables > TOML defaults) consumed by both CLI and library entry points, with a `config init` helper to scaffold templates. CLI runs should use core logging, process files serially in deterministic order, accumulate per-file errors, and offer overwrite/version controls. Deliver a shared `study init` (and optional `study config`) flow so workspace bootstrapping is decoupled from RAG-specific commands by introducing a reusable `study_utils.core.workspace.ensure_workspace` helper.
- Assumptions / Open questions
  - Workspace root will mirror existing data-home pattern (e.g., `~/.study-utils-data`) and should reuse helpers via a new shared bootstrap/resolver module adopted by RAG and convert-markdown alike.
  - Markitdown support spans TXT/PDF/DOCX/HTML; EPUB remains handled via `unstructured.partition` → Markdown serialization.
  - YAML front matter minimal schema: source path, conversion timestamp (UTC ISO-8601), original mtime; additional metadata TBD unless requested later.
- Risks & mitigations
  - Markitdown performance or crashes on malformed PDFs → isolate conversion per file, catch exceptions, record failure without halting the overall run.
  - Versioning scheme conflicts (e.g., existing `-01`) → implement deterministic increment scan to avoid overwriting.
  - Config precedence drift between CLI and library usage → bake CLI > environment variables > TOML defaults into the config loader and cover with tests.
  - Config drift or missing template → ensure loader validates schema, surfaces actionable errors, and `config init` refuses to overwrite unless `--force`.
  - Shared workspace bootstrap diverges from existing tooling expectations → coordinate updates so `study init` provisions templates for RAG and convert-markdown, with clear migration messaging.
  - New dependency (`markitdown`) increases install footprint → document requirement and gate usage with graceful error if missing.
  - Optional dependencies absent at runtime → guard markitdown/unstructured imports so the CLI surfaces actionable install guidance instead of stack traces, and cover with tests.

## Resources
### Project docs
- agents/tasks/feat/0007-document-to-markdown/spec.md — feature contract and constraints (markitdown usage, config init, shared workspace bootstrap).
- agents/guides/workflow.md — collaboration steps; ensures implementation log checklist matches new workflow reminder.
- agents/guides/engineering-guide.md — DI and seam guidance for converter + worker orchestration.
- agents/guides/patterns-and-architecture.md — module layout and logging practices to align with existing CLI tools.
- agents/guides/styleguides.md — coding standards, Ruff expectations, pytest patterns for 100% coverage.
### External docs
- https://github.com/microsoft/markitdown — API reference for MarkItDown conversion helpers.
- https://unstructured-io.github.io/unstructured/core/partition.html — EPUB parsing guidance and Markdown serialization.

## Impact Analysis
### Affected behaviors & tests
- Conversion CLI happy path → new integration tests ensuring TXT/PDF/DOCX/HTML go through markitdown with YAML front matter.
- EPUB conversion fallback → unit test around `unstructured` path with fixture EPUB.
- Workspace output + naming policy → tests for default directory, overwrite skip, overwrite true, version suffix generation.
- Error accumulation → test verifying multiple failures captured and exit code reflects count.
- Config init → test for template creation, `--force` behavior, and schema validation.
- Shared workspace bootstrap → tests (or integration coverage) ensuring `study init` provisions required directories/templates and remains compatible with RAG flows.
- Backwards compatibility explicitly not required; remove or refactor legacy helpers/tests that no longer serve the streamlined workflow.

### Affected source files
- Create: `src/study_utils/convert_markdown/__init__.py` (package), `src/study_utils/convert_markdown/cli.py`, `src/study_utils/convert_markdown/config.py`, `src/study_utils/convert_markdown/converter.py`, `src/study_utils/convert_markdown/output.py`, `src/study_utils/convert_markdown/template.toml` (packaged resource), plus helpers for sequential processing and result reporting, and a shared `src/study_utils/core/config_templates.py` that exposes template-loading utilities.
- Modify: `src/study_utils/cli.py` to register the new command; `src/study_utils/core/__init__.py` (exports if needed); `pyproject.toml` to add `markitdown` and package data; docs/README as needed; introduce or repoint shared workspace bootstrap (e.g., `src/study_utils/core/workspace.py`) and update RAG CLI/config wiring to use it.
- Delete: None expected.
- Config/flags: New `convert_markdown.toml`; environment knob for workspace root if reusing data-home env var (e.g., `STUDY_UTILS_DATA_HOME`).
### Security considerations
- Handle local files only; ensure we respect user permissions and avoid leaking contents in logs.
- When writing outputs/configs, enforce user-only permissions (align with existing data-home practices).
- Validate output paths to avoid directory traversal when honoring config-provided directories.

## Solution Plan
- Architecture/pattern choices: follow CLI module conventions (`study_utils/<feature>/cli.py` registering Typer/argparse). Keep pure conversion helpers in `converter.py`, orchestrate IO in CLI. Use `patterns-and-architecture.md` recommendation for composition, injecting converter + storage seams while keeping execution serial for clarity. Centralize packaged template access via the shared `study_utils.core.config_templates` helper so all commands pull from a single registry of bundled templates.
- DI & boundaries: expose top-level `convert_documents(options)` accepting dependencies (markitdown instance, logging hook, time provider) for easier testing. Config loader returns a dataclass consumed by CLI and enforces precedence (CLI overrides > environment variables > TOML defaults). Sequential executor consumes file list, sorts normalized paths before processing, and invokes the converter with deterministic logging. Keep YAML front matter assembly isolated in `output.py` so converter backends stay pure.
- Stepwise checklist:
  - [x] Introduce shared workspace bootstrap module and wire new `study init` (plus `study config` if warranted), landing the helper + verification tests first, then repointing existing RAG commands once the seam is proven. (Implemented `study_utils.core.workspace`, added `study init`, and migrated RAG tooling/tests.)
  - [x] Scaffold `convert_markdown` package with config loader, workspace resolver hook, and CLI skeleton registered in `study_utils.cli`.
  - [x] Implement conversion pipeline (markitdown path, EPUB fallback, YAML front matter injection, version/overwrite handling) with unit tests and ensure pytest coverage at 100%.
  - [x] Implement sequential executor and result aggregation, wire CLI summary output, and confirm Ruff + pytest (100% coverage) stay green.
  - [x] Add `config init` command using packaged TOML template, update package data + documentation, and regenerate coverage + Ruff checks.
  - [ ] Update user docs/README, finalize integration tests for CLI flows (including `study init`), run full test suite (100% coverage) and Ruff one last time before completion.
  - [ ] Confirm pytest coverage remains at 100% and Ruff reports no violations (final verification per workflow reminder).

## Test Plan
- Unit: converter behavior per format, YAML front matter construction, versioning helper, config parsing/validation, sequential result aggregation (mock markitdown/unstructured).
- Integration: CLI invocation via `study_utils.cli` covering conversion success, overwrite skip, versioning, error accumulation, config overrides, `study init`, and `config init` scaffolding.
- Contract: lightweight smoke ensuring markitdown import is present; optionally mark test xfail if library missing to keep CI clear.
- Manual checks: run CLI on sample directory with mix of supported formats; inspect generated Markdown and front matter; verify logs in workspace.

## Operability
- Logging: hook into `study_utils.core.logging` with INFO summaries (converted/skipped/failed counts) and DEBUG per-file traces when enabled.
- Telemetry/metrics: none beyond logs; ensure exit codes reflect failure counts for automation.
- Runbooks: document in README/agents spec how to run `study init`, reset workspace directories, rerun `config init`, and interpret versioned filenames; add revert notes (remove CLI registration, delete package directory, drop dependency).

## History
### 2025-03-17 01:05
**Summary**
Drafted implementation plan for Document to Markdown converter.
**Changes**
- Created initial implementation.md outlining architecture, checklist, and test strategy.

### 2025-03-17 02:05
**Summary**
Adjusted implementation plan for shared workspace bootstrap and serial execution.
**Changes**
- Added `study init`/`study config` workstream and shared bootstrap risks.
- Removed multiprocessing references in favor of deterministic serial processing.
- Updated solution/test plans and source impact to include workspace refactor and sequential executor.

### 2025-03-18 09:15
**Summary**
Implemented shared workspace helper and CLI bootstrap.
**Changes**
- Added `study_utils.core.workspace` with reusable layout resolver and fallback handling.
- Introduced `study init` CLI (`study_utils.workspace.cli`) and registered it with `study` entry point.
- Migrated RAG data-dir helpers to the shared workspace module and added unit/CLI coverage ensuring 100% test pass.

### 2025-03-18 11:20
**Summary**
Introduced convert-markdown scaffolding with shared configuration loader.
**Changes**
- Added `study_utils.convert_markdown` package with config loader, CLI skeleton, and public exports.
- Registered `study convert-markdown` command in `src/study_utils/cli.py` using the new scaffolding.
- Created comprehensive unit tests for configuration precedence/error paths and CLI behavior, keeping pytest coverage at 100%.
### 2025-03-19 09:30
**Summary**
Implemented conversion pipeline with front matter and collision handling.
**Changes**
- Added `converter.py` and `output.py` with markitdown/unstructured seams, YAML front matter serialization, and collision policy support (skip/overwrite/version).
- Exported pipeline APIs via `study_utils.convert_markdown.__init__` and wired helper constants.
- Added exhaustive unit coverage for conversion success, skips, failures, timestamp normalization, and collision edge cases; verified Ruff and pytest (100% coverage).

### 2025-03-19 13:45
**Summary**
Delivered sequential executor with CLI wiring and aggregation summary.
**Changes**
- Added `executor.py` to process sorted inputs, aggregate outcomes, and log structured status updates.
- Replaced CLI scaffolding with real execution path hooking logger setup, dependency seams, and summary output.
- Expanded test suite with executor coverage (directory walking, collision skips, fallback paths) plus updated CLI tests; re-ran Ruff and full pytest to confirm 100% coverage.

### 2025-03-19 16:05
**Summary**
Implemented config init scaffolding and template packaging for convert-markdown.
**Changes**
- Added shared `core.config_templates` registry and bundled `convert_markdown` TOML template.
- Wired `study convert-markdown config init` with workspace-aware path resolution, updated README, and included template in package data.
- Extended unit tests for template helpers and CLI error paths; re-ran Ruff and pytest (100% coverage) before completion.
