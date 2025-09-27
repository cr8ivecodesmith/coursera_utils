# CLI consolidation regressions — Implementation (Mini)

## Understanding
- Multiple `study` subcommands still rely on thin wrappers that assumed standalone execution; after consolidation their signatures drifted from the shared helpers, producing runtime errors like `parse_extensions(... default=...)`.
- Risk: other wrappers (e.g. text combiner, markdown-to-pdf, quizzer tooling) could have similar latent mismatches that tests miss because they patch helpers or exercise lower-level APIs instead of the CLI entrypoints.
- No backward compatibility with standalone scripts is required, so legacy branches and their dedicated tests can be culled while keeping the consolidated CLI behavior correct and fully covered.
- Assumption: command coverage in pytest focuses on modules, so adding CLI-focused tests and tightening lint/coverage gates (including Ruff) is necessary to pin these defaults.

## Impact
- Files / modules touched: `src/study_utils/cli.py`, `src/study_utils/generate_document.py`, `src/study_utils/text_combiner.py`, `src/study_utils/markdown_to_pdf.py`, `src/study_utils/quizzer/utils.py`, and any other subcommand wrappers using `parse_extensions` or similar helper pass-throughs.
- Remove obsolete tests/code supporting standalone invocations, while ensuring consolidated CLI paths hit 100% coverage.
- Tests to add or update: expand CLI integration coverage in `tests/test_cli.py` plus targeted unit tests for wrapper defaults to reflect consolidated behavior.
- Tooling: run Ruff linting and address any violations introduced or exposed by the cleanup.

## Plan
- [x] Inventory every `study` subcommand wrapper and document helper signatures/options they forward.
- [x] Align wrapper defaults with the shared core helpers or inline the normalization to remove redundant wrappers, dropping legacy-only branches.
- [x] Remove obsolete or redundant tests tied to standalone scripts, replacing them with CLI-focused coverage where needed.
- [x] Update CLI argument parsing to surface consistent help text and prevent positional/flag drift.
- [x] Add integration-style pytest cases that invoke `study <subcommand>` with representative arguments to catch regressions and maintain 100% coverage.
- [x] Run Ruff and pytest (with coverage gate) to confirm the cleanup passes existing quality bars.

## Tests
- [x] Unit (`pytest`)
- [x] Integration (CLI smoke cases via `pytest`)
- [x] Ruff (`ruff check`)

## History
### 2025-09-27 04:15
**Summary**
- Drafted implementation plan outlining audit/fix steps and test additions for CLI consolidation regressions.

### 2025-09-27 04:17
**Summary**
- Updated plan to drop legacy compatibility, prune useless tests, and include Ruff/coverage requirements per latest guidance.

### 2025-09-27 05:05
**Summary**
- Removed legacy import fallbacks across generate-document, markdown-to-pdf, and quizzer utilities so subcommands share the core helpers.
- Trimmed redundant tests tied to standalone scripts while adding CLI-level smoke coverage in `tests/test_cli.py` for generate-document, text-combiner, and markdown-to-pdf.
- Ran `pytest` (100% coverage) and `ruff check` to confirm the cleanup passes existing quality gates.
