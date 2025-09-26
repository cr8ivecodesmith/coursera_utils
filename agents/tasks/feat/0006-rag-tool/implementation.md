# Study RAG Tool — Implementation

## Understanding
- Build a `study rag` CLI surface that can ingest study materials into portable, named vector DBs, manage them (list/export/import/delete), and drive a `rich` chat client that resumes sessions using those DBs. All artifacts live under `~/.study-utils-data`, including configs, logs, DBs, and chat sessions.
- Assumptions / Open questions
  - Launch scope targets OpenAI only; design adapters so additional providers can plug in later without touching CLI surfaces.
  - Default chunking uses `tiktoken` (300-token chunks, 30-token overlaps) with config overrides and a fallback whitespace splitter when tokenizer load fails.
  - Ingestion deduplicates documents by checksum; need manifest fields to track versioning for repeated ingests.
- Risks & mitigations
  - Large local corpora could bloat disk usage → include disk usage reporting and guardrail messaging.
  - Config drift or corrupt manifests → implement schema validation + checksum metadata for vector stores.
  - Session resumes relying on missing DBs → detect and prompt user to reattach or exit gracefully.
  - Concurrent CLI runs could corrupt manifests or sessions → rely on atomic writes (`tempfile` + `os.replace`) and per-resource locks.
  - Importing external archives poses traversal risks → normalize members, reject unsafe paths, and re-verify manifest checksums post-import.

## Resources
### Project docs
- agents/tasks/feat/0006-rag-tool/spec.md — feature contract for behavior and constraints.
- agents/guides/engineering-guide.md — guidance on seams, DI, and testing defaults.
- agents/guides/patterns-and-architecture.md — CLI module layout, logging, and composition guidance.
- agents/guides/styleguides.md — Ruff lint expectations, docstring style, testing patterns.
### External docs
- https://python.langchain.com/docs/modules/data_connection/vectorstores/how_to/faiss — reference for FAISS vector store usage.
- https://rich.readthedocs.io/en/stable/introduction.html — CLI rendering patterns for interactive chat UI.
- https://docs.python.org/3/library/tomllib.html — stdlib TOML parsing for config ingestion.

## Impact Analysis
### Affected behaviors & tests
- `study rag ingest` → new integration tests using temp dirs validating manifests and FAISS index creation.
- `study rag list/export/import/delete` → unit tests around repository manager logic, manifest metadata (tokenizer, embedding, chunking), and CLI parsing.
- `study rag chat` → integration test driving a mock LLM and ensuring session persistence / resume.
- `study rag config` → unit tests for template generation, precedence (CLI > env > config), validation, and error messaging.
### Affected source files
- Create: `src/study_utils/rag/__init__.py`, `src/study_utils/rag/config.py`, `src/study_utils/rag/data_dir.py`, `src/study_utils/rag/ingest.py`, `src/study_utils/rag/vector_store.py`, `src/study_utils/rag/chat.py`, `src/study_utils/rag/session.py`, `src/study_utils/rag/cli.py`, plus package resources for config template.
- Modify: `src/study_utils/cli.py` to register `rag` subcommands; `pyproject.toml` if extra dependencies arise; documentation files (`README.md`, `agents/tasks/...`).
- Delete: None anticipated.
- Config/flags: New TOML template `~/.study-utils-data/config/rag.toml`; env var override (e.g., `STUDY_UTILS_DATA_HOME`).
### Security considerations
- Ensure file permissions set to user-only when creating config, DB, session directories.
- Avoid logging raw user documents or chat messages unless verbose/debug explicitly enabled.
- Validate inputs for archive import to mitigate path traversal; prefer safe extraction routines.

## Solution Plan
- Favor composition: separate ingestion pipeline, vector store abstraction, and chat runtime per `patterns-and-architecture.md` guidance.
- Leverage dependency injection seams from `engineering-guide.md` by passing provider adapters and storage paths explicitly (facilitates testing with in-memory mocks).
- Stepwise checklist:
  - [x] Milestone 1 — Data home + config framework: implement data dir resolver, TOML loader, `study rag config` commands; working CLI build with existing tests and Ruff passing.
  - [x] Milestone 2 — Vector store ingestion & management: add chunker, embedding pipeline, DB manifest schema (embedding model, tokenizer, chunk sizes, checksums), list/export/import/delete commands with atomic writes and safe archive handling; ensure full test suite (new + existing) at 100% and Ruff clean.
  - [x] Milestone 3 — Chat runtime & sessions: build retrieval orchestration, `rich` chat loop, session persistence/resume with manifest compatibility checks and locking; verify working CLI build, all tests, Ruff lint.
  - [x] Milestone 4 — Polish & docs: finalize telemetry/logging, doctor command, docs updates, cross-platform permission handling, confirm 100% tests + Ruff, prep release notes.

## Test Plan
- Unit
  - Config loader validation (defaults, overrides, error reporting).
  - Data directory utilities (path resolution, permission setup, environment overrides).
  - Vector store manifest serializer/deserializer with checksum, embedding model, tokenizer, and chunking metadata handling.
  - Session repository (create/resume, missing DB handling, manifest compatibility checks) under mocked storage with locking.
- Integration
  - Ingestion pipeline against temp markdown/PDF fixtures producing FAISS artifacts with atomic writes verified.
  - CLI command flows using `typer/pytest` runner verifying stdout/stderr formatting and return codes.
  - Chat loop with mocked LLM + retrieval ensuring messages append and resume correctly.
- Contract
  - Provider adapter smoke tests using configured embedding/chat providers (feature-flagged or skipped if keys absent) to ensure API compatibility.
- Manual
  - Dry run on a small corpus to inspect interactive chat UX and ensure logs/configs created correctly.
  - Import/export archive sanity check ensuring traversal defenses trigger when expected.

## Operability
- Wire logging to `~/.study-utils-data/logs/rag.log` with rotation strategy; expose `--verbose` / config toggle.
- Update README with data directory overview, config schema (including precedence), command examples, and troubleshooting tips.
- Extend existing runbook with instructions for clearing caches, exporting/importing DBs, and recovering sessions.

## History
### 2025-09-30 16:30
**Summary**
Delivered Milestone 3 with the chat runtime, session persistence, and CLI integration.
**Changes**
- Added chat orchestration module with retrieval aggregation, OpenAI adapter, and rich-powered loop.
- Implemented session store with locking plus CLI `chat` command for new/resumed conversations.
- Introduced targeted tests covering sessions, runtime flows, and CLI usage while keeping coverage guidance intact.

### 2025-10-02 09:45
**Summary**
Delivered Milestone 4 with shared logging/config utilities, diagnostics, and docs.
**Changes**
- Added reusable core logging helpers and integrated JSON file logging with
  CLI overrides.
- Added the `study rag doctor` command with environment checks, dependency
  audit, and disk reports.
- Extended tests for logging and doctor flows and refreshed CLI
  documentation.

### 2025-09-29 09:30
**Summary**
Polished Milestone 2 by implementing the remaining inspect workflow.
**Changes**
- Added `study rag inspect` CLI support to display manifest metadata and document provenance.
- Extended CLI tests to cover inspect success and error messaging.

### 2025-09-28 14:10
**Summary**
Delivered Milestone 2: ingestion pipeline, vector store management commands, and supporting tests.
**Changes**
- Implemented chunking/dedup pipeline with OpenAI embedding adapter plus manifest writer.
- Added FAISS/in-memory vector store backends with repository export/import/delete capabilities.
- Extended `study rag` CLI with ingest/list/export/import/delete commands and comprehensive tests (coverage 100%).

### 2025-09-27 12:45
**Summary**
Delivered Milestone 1: data directory utilities, config loader/template, and `study rag config` CLI with full lint/test pass.
**Changes**
- Implemented data home resolver honoring `STUDY_UTILS_DATA_HOME` plus permissions setup.
- Added typed config schema, TOML loader, and template generator wired into `study rag config` commands.
- Registered `study rag` CLI entry point and added unit/CLI tests ensuring pytest coverage and Ruff success.

### 2025-09-27 00:17
**Summary**
Drafted implementation plan for Study RAG tool.
**Changes**
- Created initial implementation.md with milestones and test strategy.
