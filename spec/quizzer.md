Quizzer Spec
===

## **Sources**

- `src/study_utils/quizzer`

## **Description**

Interactive terminal quiz tool that extracts topics and multiple‑choice questions from Markdown sources, stores a local bank, runs practice sessions, and summarizes weak areas for review. AI is optional and configurable; artifacts are human‑editable.

## **Python Compatibility**

- Target: Python 3.8+.
- Optional UI: `textual` for a richer TUI; pure CLI fallback when not installed.

## **CLI**

- Base: `python -m study_utils.quizzer [command] [options]`
- Global options:
- `--config PATH`: Path to `quizzer.toml` (default: look in CWD, then `study_utils/`).
  - `--out DIR`: Artifacts directory (default: `.quizzer/<quiz_name>` under CWD).
  - `--model NAME`: Override AI model for this run.
  - `--seed INT`: Global RNG seed for deterministic selection.
  - `--dry-run`: Print plan without writing files or calling AI.
  - `--verbose`: Print detailed steps.
- Commands:
  - `init QUIZ_NAME`: Create `quizzer.toml` template for a quiz.
  - `topics generate QUIZ_NAME [--force] [--limit N] [--extensions md markdown] [--level-limit N]`: Extract and persist topics.
  - `topics list QUIZ_NAME [--filter STR]`: List known topics.
  - `questions generate QUIZ_NAME [--per-topic N] [--type mcq] [--regenerate-missing]`: Generate and validate questions with answers.
  - `questions list QUIZ_NAME [--topics TOPIC... ]`: List questions in bank.
  - `start QUIZ_NAME [--num N] [--mix balanced|random|weakness] [--topic TOPIC ...] [--resume] [--shuffle] [--explain|--no-explain] [--time-limit SEC]`: Run a quiz session.
  - `review QUIZ_NAME [--session ID]`: Re‑quiz wrong/weak topics from a prior session.
  - `report QUIZ_NAME [--session ID]`: Show session summary and per‑topic stats.

## **Configuration: `quizzer.toml`**

- Layout: multiple quiz definitions under `[quiz.<name>]`.
- Required fields under each quiz:
  - `sources = ["./path", ...]`: Files or directories (recursively discover `*.md`, `*.markdown`).
  - `types = ["mcq"]`: Supported question types; start with `mcq`.
- Recommended fields:
  - `materials = ["./refs/reading.txt", ...]`: Optional reference texts for explanations.
  - `per_topic = 3`: Default questions to generate per topic.
  - `ensure_coverage = true`: Ask each topic at least once before repeats.
  - `focus_topics = ["Topic A", ...]`: Limit generation/selection to these topics.
  - `topic_seed = 0` and `question_seed = 0`: Deterministic AI generation when set.
  - `user_background = "..."`: Used for analogies in explanations.
  - `[ai] model = "gpt-4o-mini"; temperature = 0.2; max_tokens = 600; system_prompt = "..."`.
  - `[storage] out_dir = ".quizzer/<name>"`: Override artifacts directory.

## **Artifacts & Schemas**

- Topics: `topics.jsonl` (one per line)
  - `{ "id": "slug", "name": "Topic", "description": "...", "source_paths": ["..."], "created_at": "ISO8601" }`
- Questions: `questions.jsonl` (one per line)
  - `{ "id": "uuid", "topic_id": "slug", "type": "mcq", "stem": "...", "choices": [{"key":"A","text":"..."},...], "answer": "B", "explanation": "...", "sources": ["file.md#h3"], "version": 1, "created_at": "ISO8601" }`
- Sessions: `sessions/<timestamp>/`
  - `responses.jsonl`: `{ "question_id": "...", "given": "A", "correct": true, "duration_sec": 7.2, "topic_id": "..." }`
  - `summary.json`: `{ "id": "...", "started_at": "...", "ended_at": "...", "total": 20, "correct": 16, "accuracy": 0.8, "per_topic": {"slug": {"asked": 3, "correct": 2}}, "weak_topics": ["slug"], "mix": "balanced" }`

## **Behavior**

- Topic generation:
  - Parse sources deterministically; deduplicate by normalized slug; keep short description per topic; append‑only JSONL to allow manual edits/merges.
  - Respect `focus_topics` when present; otherwise discover from content and headings.
  - Optional AI assist: when enabled, prompt a model with short snippets to propose topic names/descriptions; merge with heuristic headings by slug, prefer clearer names, and keep `source_paths`.
- Question generation:
  - Generate at least `per_topic` MCQs per topic; enforce single correct answer; plausible distractors; stem and choices concise; include brief explanation and source refs when possible.
  - Validate format (letters A–D unique, exactly one answer, non‑empty fields). Refuse to save invalid entries with actionable error messages.
- Selection strategy (`start`):
  - `balanced`: Ensure each topic appears once before repeats; use RNG with `--seed` when provided.
  - `random`: Uniform random over all questions.
  - `weakness`: Weight selection toward topics with lower accuracy in recent summaries.
  - Filters: `--topic` limits pool; `focus_topics` from config applies by default.
- Session flow:
  - Accept answers as letter (`a`/`A`) or choice text; allow `skip`, `back`, `quit`.
  - Show correctness, short explanation, and optional source snippet.
  - Track progress (e.g., `12/20 • 80%`), elapsed time, and per‑topic stats live.
  - On completion, write `responses.jsonl` and `summary.json`; print a concise report and list weak topics.
- UI:
  - If `textual` is available, present a TUI with keyboard navigation; otherwise use simple line‑oriented prompts.

## **Implementation Outline**

- Pure helpers (unit testable):
  - File discovery (`iter_quiz_files`), topic extraction, question validation, selection algorithms, scoring and summary aggregation, JSONL IO utils.
  - AI topic extraction helper `ai_extract_topics(sources, k, client, prompt, model, temperature, seed)` that returns suggested topics; merging logic lives in `extract_topics(use_ai=True, ...)`.
  - Seed control via `random.Random` instances passed explicitly.
- I/O edges and CLI (`main()`):
  - Argparse subcommands; minimal side effects; all paths via `pathlib.Path`.
  - AI calls isolated in `ai.py` with `load_client()`; easy to stub.

## **Testing**

- Unit tests with stubbed AI:
  - Topic extraction from simple Markdown inputs (headings, bullet lists)
  - Question validator catches bad keys, multiple answers, empty fields
  - Selection strategies honor coverage, weakness weighting, and seeds
  - Summary aggregation computes per‑topic accuracy correctly
- Integration‑light tests:
  - CLI parse for `init`, `topics generate`, `questions generate --per-topic 2`, `start --num 3 --mix balanced --dry-run`
  - JSONL read/write round‑trip for topics/questions

## **Error Messages & Remediation**

- Missing or invalid `quizzer.toml`: point to `init QUIZ_NAME` to scaffold.
- No sources resolved: suggest `--extensions`, `--level-limit`, or fix paths.
- Invalid question format: show precise line and reason; suggest `questions validate`.
- AI errors: explain `.env`/`OPENAI_API_KEY` support via `load_client()` and retry options.

## **Security & Privacy**

- No shell invocation; sanitize/resolve paths; never write outside `--out`.
- Offline‑friendly: all AI calls optional; tests stub external clients.

## **Technology**

- AI via OpenAI client from `load_client()`; light retries/backoff; timeouts configurable.
- Optional `textual` for TUI; otherwise standard input/print for CLI.
