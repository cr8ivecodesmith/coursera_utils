# Generate Document Config Init — Spec

## Summary
Extend the AI-powered document generator so it mirrors the convert-markdown workflow: ship a packaged TOML template, scaffold it via `study generate-document config init`, and refactor the command into a dedicated subpackage that cooperates with the shared workspace helpers.

## Goals
- Introduce a `study generate-document config` command group with an `init` subcommand that writes `documents.toml` to either an explicit `--path` or the workspace config directory, with `--workspace` and `--force` flags matching convert-markdown.
- Promote `src/study_utils/generate_document.py` into a `study_utils.generate_document` package (e.g., `cli.py`, `config.py`, `runner.py`) and delete the legacy module shim instead of preserving backward-compatible imports/tests.
- Remove obsolete CLI aliases, fixtures, and tests that only exercised the legacy entry point so the codebase stays lean.
- Register a packaged `template.toml` for generate-document with `study_utils.core.config_templates` so both `study init` and the new config command can scaffold it.
- Update config discovery so the CLI defaults to the workspace config file before falling back to CWD or bundled templates, and emit an actionable error pointing to `config init` when nothing is found.
- Refresh README/quickstart snippets and pytest coverage to reflect the new command flow, including CLI tests that cover success, overwrite protection, and workspace resolution.

## Non-Goals
- No redesign of prompt content, document types, or OpenAI parameters beyond relocating configuration.
- No changes to the existing AI client plumbing or model selection heuristics outside of respecting the relocated config.
- No new telemetry pipeline or dependency on additional third-party packages.
- No attempt to batch or parallelize generation; the core `generate_document` behavior stays single-shot per invocation.

## Behavior (BDD-ish)
- Given a fresh workspace and no `documents.toml`, when the user runs `study generate-document config init`, then the command ensures the workspace exists, writes the packaged template into `<workspace>/config/documents.toml`, and prints the destination path.
- Given a custom destination file that already exists without `--force`, when the user runs `study generate-document config init --path ./documents.toml`, then the command exits non-zero with a message explaining that the file exists and how to use `--force`.
- Given an existing config and `--force`, when the user reruns the init command, then the template overwrites the target and the CLI reports success.
- Given a scaffolded config in the workspace, when the user runs `study generate-document keywords out.md src/`, then the CLI resolves the workspace config (unless `--config` overrides it), loads prompts from the TOML, and generates output as before.
- Given no config found anywhere, when the user runs the generate-document CLI, then it fails before invoking the OpenAI client and directs the user to run `study generate-document config init`.

## Constraints & Dependencies
- Constraints: maintain ASCII templates and docs; keep the CLI and helper modules pure/testable; continue targeting Python 3.11+ for `tomllib` while preserving the `tomli` fallback for older environments in tests.
- Dependencies: reuse `study_utils.core.workspace.ensure_workspace`, `study_utils.core.config_templates`, and existing OpenAI + file helpers; update `pyproject.toml` (and package data) so the new template ships with the wheel; mirror convert-markdown’s CLI ergonomics for consistency; ensure `study init` installs the new template via the shared template registry.
- Upstream/Downstream: update `study_utils.cli` command wiring, pytest fixtures that import `study_utils.generate_document`, and any tooling that assumed the template lived beside `generate_document.py`.

## Security & Privacy
- Continue to read and write configs locally; do not log prompt contents or document excerpts; ensure generated configs default to restrictive file permissions (0o600) via the shared TOML writer.
- Preserve existing environment-variable handling for API keys and avoid introducing new secret stores.

## Telemetry & Operability
- Maintain existing stdout messaging for successful generation; optionally add structured logging hooks only if needed for config errors.
- Document the new command in README and mention how to inspect workspace locations for troubleshooting.

## Rollout / Revert
- Rollout: land the package refactor, template registration, CLI surface, docs, and tests together; verify `study init` now installs both convert-markdown and generate-document configs.
- Revert: remove the new package modules, drop the template entry, restore the standalone `generate_document.py`, and excise command wiring changes.

## Definition of Done
- [ ] Behavior verified
- [ ] Docs updated (user/dev/ops)
- [ ] Tests added/updated (unit/contract/integration/e2e)
- [ ] Flags defaulted per channel (dev/beta/stable)
- [ ] Monitoring in place

## Ownership
- Owner: @matt
- Reviewers: @codex
- Stakeholders: @matt

## Links
- Related: src/study_utils/generate_document.py, src/study_utils/cli.py, src/study_utils/core/config_templates.py, agents/tasks/feat/0007-document-to-markdown/spec.md

## History
### 2025-03-30 10:00
**Summary** — Draft spec for generate-document config scaffolding
**Changes**
- Captured goals, behaviors, and rollout plan for adding `study generate-document config init` and the supporting refactor.
### 2025-09-30 13:39
**Summary** — Drop backwards-compatibility requirement
**Changes**
- Clarified that the generate-document refactor removes the legacy module shim and deletes redundant tests/aliases instead of keeping compatibility layers.
