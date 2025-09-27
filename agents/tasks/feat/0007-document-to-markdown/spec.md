# Document to Markdown Converter — Spec

## Summary
Deliver a CLI and library workflow that turns supported documents (PDF, DOCX, HTML, TXT, etc.) into Markdown files while preserving each source file’s basename, with behavior tuned via a dedicated TOML config. Introduce a unified workspace bootstrap command so converters and other tools share the same initialization path.

## Goals
- Provide `study convert-markdown` CLI for file/folder conversion to `.md` outputs.
- Add a top-level `study init` command (and supporting `study config` flow if needed) that bootstraps the shared workspace directory and installs default config templates for convert-markdown and other tools through a shared `study_utils.core.workspace.ensure_workspace` helper (idempotent and reusable by RAG tooling), rolling out in two phases (introduce helper + tests, then re-point existing commands) so current RAG flows stay stable.
- Preserve each input filename (e.g., `notes.pdf` → `notes.md`) and default outputs under a workspace directory (e.g., `~/.study-utils-data/converted`), with CLI/config options for alternate destinations.
- Centralize converter settings (extensions, output location, overwrite/versioning policy, backend knobs) in a new `convert_markdown.toml`.
- Use Microsoft’s `markitdown` library for structure-preserving conversions of TXT/PDF/DOCX/HTML, while delegating EPUB handling to `unstructured`.
- Expose a reusable library API so other modules can call the converter programmatically.
- Emit YAML front matter capturing basic metadata (source path, conversion timestamp, original modified time) ahead of the Markdown body.
- Provide `study convert-markdown config init` to scaffold a template `convert_markdown.toml` with documented defaults.

## Non-Goals
- No Markdown → other format conversion (existing `markdown_to_pdf` covers downstream needs).
- No GUI/TUI; keep scope to CLI and importable functions.
- No bespoke OCR/image text extraction beyond what `unstructured` already provides.
- No automated cleanup or summarization of converted content; focus on faithful Markdown output.
- No obligation to preserve existing behaviors/tests that become obsolete; prune legacy code where it adds bloat.

## Behavior (BDD-ish)
- Given a DOCX file and defaults, when the user runs `study convert-markdown source.docx`, then `source.md` is written under the workspace directory with headings, lists, and paragraphs preserved by `markitdown` plus YAML front matter.
- Given a directory and `--extensions pdf docx`, when the user runs the command, then only matching files are converted and each produces a Markdown file mirroring its basename.
- Given an existing Markdown output and overwrite disabled via config, when the user reruns the converter, then the file is skipped and reported in the summary.
- Given `--overwrite` or `--version-output`, when name collisions occur, then files are either replaced or versioned with `-01`, `-02`, etc. according to the selected policy.
- Given an unsupported extension, when conversion is attempted, then the file is skipped with an explanatory log, the summary reports accumulated failures, and the command exits non-zero if any conversion fails.
- Given `--config custom.toml`, when the command runs, then options (output location, overwrite behavior, logging) come from that TOML file, with precedence enforced as CLI flags > environment variables > TOML defaults via a centralized loader shared by the CLI and library surfaces.
- Given EPUB input, when conversion runs, then the EPUB is parsed via `unstructured` and rendered to Markdown while other formats continue to use `markitdown`.
- Given multiple inputs, when the user runs the command, then files are converted sequentially in a deterministic, path-sorted order while still producing a per-file success/failure summary.
- Given `study init` with no workspace present, when the user runs the command, then the shared workspace directory is created and default config templates (including `convert_markdown.toml`) are installed without relying on tool-specific bootstrappers, leveraging the shared workspace helper to remain idempotent.
- Given no existing config, when the user runs `study convert-markdown config init`, then a TOML template is created at the requested path (or default workspace config location) containing documented keys and default values.

## Open Questions
- None at this time; design decisions integrated from stakeholder feedback above.

## Constraints & Dependencies
- Constraints: must respect existing coding standards (80-char lines, pure helpers + isolated I/O). Converter should operate offline; no network calls. YAML front matter assembly must happen in a dedicated serializer so conversion backends stay pure.
- Dependencies: `markitdown` for TXT/PDF/DOCX/HTML conversion; `unstructured` for EPUB fallback; reuse `study_utils.core` helpers (extension parsing, logging, config loaders) and the core logging module for structured output; rely on `tomllib` for config parsing; share a centralized workspace/bootstrap utility (`study_utils.core.workspace.ensure_workspace`) invoked by both `study init` and other CLI commands; leverage existing config-init patterns from the RAG tool while decoupling them from RAG-specific entry points; expose a single importlib-resources helper (`study_utils.core.config_templates`) to access bundled config templates so `study init` and `study convert-markdown config init` stay in sync, with the helper acting as the single registry for packaged templates; guard optional converter imports so missing dependencies surface as actionable CLI errors instead of stack traces.
- Upstream/Downstream: integrate with `study_utils.cli` for command registration; ensure new config file is packaged (update `pyproject.toml` data list) for downstream modules.

## Security & Privacy
- Process local files only; avoid transmitting contents externally.
- Sanitize logging to exclude sensitive file content—log file paths/status instead of snippets.
- Maintain filesystem permissions consistent with existing tooling (respect user umask; avoid widening access).

## Telemetry & Operability
- Emit structured logs via existing logging utilities summarizing converted/skipped/failed counts.
- Provide exit codes conveying success (0), partial success (1?), or fatal errors (>1) for automation.
- Consider optional verbose flag for detailed per-element info to aid debugging complex documents.

## Rollout / Revert
- Roll out by adding the new CLI command and module; no feature flag expected.
- Revert by unregistering the command, removing the module/config references, and deleting package data entry; no data migrations required.

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
- Related: agents/tasks/feat/0002-markdown-to-pdf/spec.md

## History
### 2025-03-17 00:00
**Summary** — Draft spec for Document to Markdown converter
**Changes**
- Created initial feature spec under agents/tasks/feat/0007-document-to-markdown/spec.md.

### 2025-03-17 00:45
**Summary** — Integrated stakeholder feedback
**Changes**
- Switched planned converter backend to `markitdown` with EPUB handled via `unstructured`.
- Defined default workspace output, versioning policy, front matter requirements, multiprocessing defaults, and error handling.
- Documented dependency on core logging module and clarified lack of open questions.

### 2025-03-17 00:55
**Summary** — Added config template requirement
**Changes**
- Added CLI goal/behavior for `study convert-markdown config init` to scaffold TOML templates.
- Noted reuse of RAG config-init pattern within dependencies.

### 2025-03-17 01:10
**Summary** — Clarified lack of backward compatibility
**Changes**
- Documented non-goal to remove obsolete behaviors/tests rather than maintaining legacy compatibility.

### 2025-03-17 02:00
**Summary** — Added shared workspace bootstrap and serial flow
**Changes**
- Introduced `study init` (+ optional `study config`) goal for unified workspace setup.
- Replaced multiprocessing requirement with deterministic sequential processing behavior.
- Updated dependencies to reference a shared bootstrap utility instead of RAG-specific wiring.
