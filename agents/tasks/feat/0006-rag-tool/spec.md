# Study RAG Tool — Spec

## Summary
Introduce a `study rag` CLI workspace that can index arbitrary study materials into named, portable vector databases and drive retrieval-augmented chat sessions that can be paused and resumed across machines.

## Goals
- Build an ingestion workflow that chunks/parses files from user-specified paths, generates embeddings, and writes them into named vector stores under `~/.study-utils-data/rag_dbs/<db-name>`.
- Allow multiple vector store backends (start with FAISS + metadata manifest) while keeping a pluggable abstraction for future providers.
- Provide management commands to list, inspect, export/import, and delete vector stores while preserving source metadata for reproducibility.
- Deliver a `rich`-powered CLI chat that can use one or more selected vector stores simultaneously, with session transcripts, prompts, and model configuration saved under `~/.study-utils-data/rag_sessions/<session-id>`.
- Support resuming prior chat sessions, continuing with the same context, and optionally switching/adding vector stores mid-session.
- Default to checksum-based deduplication during ingestion with `tiktoken`-based chunking tuned to 300-token windows and 30-token overlaps, while allowing config-driven overrides and capturing tokenizer/embedding/chunking metadata in manifests for reproducibility.
- Expose ingestion, retrieval, prompting, and token limit settings through a TOML config (`~/.study-utils-data/config/rag.toml`) with clear schema and per-command overrides.
- Provide a `study rag config` command surface to scaffold a template config, validate existing configs, and document supported keys.
- Centralize all tool artifacts (vector DBs, sessions, config, logs) inside `~/.study-utils-data` with sensible defaults and user overrides via env vars or CLI flags.

## Non-Goals
- Building a graphical UI or web service; the scope is CLI only.
- Implementing automatic continuous ingestion/watchers; ingestion is triggered manually.
- Shipping proprietary model weights or managing remote-hosted databases.

## Behavior (BDD-ish)
- Given source markdown/PDF files and a desired name, when the user runs `study rag ingest --name physics-notes path/to/files`, then the tool creates `~/.study-utils-data/rag_dbs/physics-notes` with embeddings, manifests, and reports success.
- Given multiple existing vector DBs, when the user runs `study rag list`, then the CLI prints the DB names, creation timestamps, source summary, and embedding model info.
- Given a previous ingestion, when the user runs `study rag export --name physics-notes --out physics.zip`, then a portable archive containing the FAISS index, manifest, and source metadata is created.
- Given one or more DB names, when the user runs `study rag chat --db physics-notes --db calc`, then the CLI opens a `rich` chat, routes retrieval against both DBs, streams answers, and writes/updates a session folder.
- Given a saved session ID, when the user runs `study rag chat --resume <session-id>`, then the CLI reloads conversation history, validates referenced DB manifests/tokenizer compatibility, restores the associated DB list and model settings, and appends new turns or guides the user to reconcile missing assets.
- Given no existing config, when the user runs `study rag config init`, then a TOML file is created in `~/.study-utils-data/config/rag.toml` populated with recommended defaults and inline comments.
- Given a populated config, when the user runs `study rag config validate`, then the CLI checks for unknown keys, conflicting values, and outputs actionable errors.
- Given overlapping configuration via CLI flags, environment variables, and TOML entries, when precedence is resolved, then CLI overrides env, env overrides config, and validation surfaces ambiguous or unsupported keys with helpful guidance.

## Constraints & Dependencies
- Constraints: must operate offline after embeddings are generated; prefer existing dependencies (`langchain`, `faiss-cpu`, `rich`). Respect 80-char line length in code. Config parsing should rely on stdlib (`tomllib` for Python 3.11+) or a minimal dependency. Chunking should use `tiktoken` tokenization by default with configurable chunk/overlap sizes and a fallback splitter if tokenizer initialization fails. File writes for manifests, indexes, and sessions should be atomic or guarded by advisory locks to withstand concurrent invocations and crashes.
- Upstream/Downstream: Uses OpenAI for embeddings and chat at launch via existing env conventions (`OPENAI_API_KEY`). Future multi-provider support should stay isolated behind adapters. Config must support swapping providers, embedding models, tokenizer settings, and precedence rules without code changes.

## Security & Privacy
- Keep all artifacts local; never transmit source documents or chat transcripts unless the chosen LLM provider requires it (document in warnings).
- Store API keys only in env vars/dotenv, not in manifests. Ensure session metadata includes provider name but redacts secrets.
- Set created directories/files to user-only permissions (`0o700` for directories, `0o600` for manifest/session files) where the OS allows it.
- Validate archive imports with path normalization, reject traversal attempts, and verify manifest checksums before accepting ingested data.

## Telemetry & Operability
- Emit structured CLI logs to `~/.study-utils-data/logs/rag.log` at INFO by default, DEBUG with `--verbose`.
- Record ingestion summaries (document counts, token usage, failures) in each DB manifest and optionally print to stdout.
- Provide a `study rag doctor` command to validate the data directory, list missing dependencies, report config precedence issues, verify tokenizer availability, and show disk usage per DB/session.

## Rollout / Revert
- Introduce new `study rag` subcommands; no feature flag. Document manual rollback by deleting `rag_*` directories and removing CLI entrypoints.
- Migration: create the `~/.study-utils-data` tree on first use; ensure existing commands tolerate directory absence.

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
- Related: agents/tasks/feat/0005-study-cli/spec.md

## History
### 2025-09-27 00:07
**Summary** — Draft spec for Study RAG Tool
**Changes**
- Created initial feature spec.
