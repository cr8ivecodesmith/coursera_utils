Coursera Utils — Agent Guidelines
=================================

This document defines how agents and contributors should work in this
repository. It expands the core principles to include concrete, actionable
guidance for design, coding, testing, and collaboration.


Core Expertise & Principles
---------------------------
- Python-first: You are a Python programming expert.
- Full‑stack perspective: Comfortable across CLI, scripts, APIs, and light UX.
- Best practices: Favor clarity, explicitness, and maintainability.
- Small units: Prefer small, easily testable functions. Always ask:
  “How would I test this?”
- For humans: Follow SOLID and the Zen of Python; write readable code and
  precise docstrings.
- Pragmatic decisions: Use first principles and optimize for leverage and
  simplicity over cleverness.


Coding Standards
----------------
- Typing: Add type hints to public functions and non-trivial internals.
- Docstrings: Use concise docstrings with purpose, parameters, and returns.
- Names: Choose descriptive names; avoid one-letter variables outside small
  scopes or indices.
- Structure: Prefer pure functions; pass dependencies explicitly. Avoid global
  state.
- Errors: Raise precise exceptions; include actionable messages. Don’t bury
  errors—surface them early.
- Filesystem: Use `pathlib.Path`; sanitize filenames for cross-platform safety.
- CLI: Use `argparse` with helpful `--help`, safe defaults, and clear errors.
- Performance: Keep algorithms simple; avoid accidental O(n^2). Stream or chunk
  large I/O.
- Side effects: Isolate I/O at the edges; keep core logic side‑effect free to
  simplify testing.


Testing Philosophy
------------------
- Framework: Use `pytest`. Tests live under `tests/` (see `pytest.ini`).
- Scope: Start with unit tests for the smallest units you write or change; add
  integration tests sparingly when necessary.
- Determinism: Avoid sleeping/network/file nondeterminism in unit tests; mock
  external APIs and I/O. See `tests/conftest.py` for patterns.
- Boundaries: Test parsing/formatting, path logic, name generation, error
  conditions, and cache upgrade paths.
- Speed: Keep tests fast. Use temp dirs (`tmp_path`) and avoid large fixtures.
- Coverage target: Maintain or improve coverage for changed modules (see
  `pytest.ini` which runs `--cov=study_utils`).


Tooling & Commands
------------------
- Install (dev): `pip install -r requirements_dev.txt`
- Lint: run `ruff` and `flake8` on changed files only.
  - Example: `ruff check src/study_utils/transcribe_video.py` and `flake8 src/study_utils/transcribe_video.py`
- Format: use `black` on changed files.
  - Example: `black src/study_utils/transcribe_video.py`
- Tests: `pytest -q` (configured with `-n auto` and coverage in `pytest.ini`).
- Fast search: prefer `rg` (ripgrep) for code search over `grep`.


Running Tests (practical tips)
------------------------------
- Default: `pytest -q` (parallel via `-n auto`, coverage on `study_utils/`).
- Subset: `pytest -q -k text_combiner` or `pytest tests/test_text_combiner.py -q`.
- Collection debug: `pytest --collect-only -q` to spot import/path issues early.
- Bypass repo config (debug): `pytest -q -c /dev/null tests/test_text_combiner.py` to ignore `pytest.ini` when isolating failures.
- Package layout: modules live under `src/study_utils/` and can be run as
  `python -m study_utils.transcribe_video` or `python -m study_utils.text_combiner`
  once the project is installed (e.g., `uv pip install -e .`).
- Heavy deps are stubbed in `tests/conftest.py` (e.g., `openai`, `pydub`), keeping unit tests fast and offline.


Repository Conventions
----------------------
- Python: Prefer `pathlib`, `datetime` with `timezone`, and stdlib options.
- Env/config: Read configuration from environment variables; support `.env`
  via `python-dotenv` (see `load_client()` in `transcribe_video.py`).
- External APIs: Add light retries/backoff around network calls if needed; keep
  timeouts conservative and configurable.
- File writes: Be explicit where files are written. Avoid destructive defaults;
  prefer creating or overwriting within an output dir specified by the user.
- Logging vs print: For CLI tools, `print` is acceptable for progress and
  results. Keep messages concise and informative.


Design Guidance (This Repo)
---------------------------
- Discovery: Keep path discovery functions pure and well-typed; do not touch
  the FS beyond reading.
- Naming: Isolate “smart name” heuristics in small, composable functions; keep
  AI-assisted naming optional, resilient, and sanitized.
- Caching: Persist human-editable JSON where appropriate. Maintain versioned
  cache schemas and provide upgrade paths.
- Chunking: For audio/video processing, keep chunk sizes and formats
  configurable and documented.
- Prefixing: Treat CLI prefix composition as a pure transformation; ensure
  deterministic output given inputs.


Security & Privacy
------------------
- Secrets: Never hardcode API keys. Expect `OPENAI_API_KEY` via env or `.env`.
- Inputs: Sanitize user input used in paths or filenames; avoid invoking shell
  commands with user-provided strings.
- Network: Fail fast and clearly on network/API errors; provide actionable
  remediation (“set OPENAI_API_KEY…”, “check connectivity…”, etc.).
- Data: Write outputs only to user-specified or clearly documented locations.


When Modifying Code
-------------------
- Narrow scope: Change only what’s necessary to implement the task.
- Preserve behavior: Avoid breaking existing CLI flags and outputs unless the
  change is intentional and documented.
- Tests first: Add or update tests alongside the change when feasible.
- Backwards compatibility: Treat legacy flags/fields as deprecated with clear
  handling and comments (e.g., treat `sep:` as `text:` as in tests).
- Review checklist:
  - Clear types and docstrings
  - Happy path + edge cases covered by tests
  - No unexpected FS or network side effects
  - Lint/format pass cleanly


Contributor Workflow (for agents)
---------------------------------
- Read before writing: Skim `README.md`, `transcribe_video.py`, and `tests/` to
  align with existing patterns and expectations.
- Plan small: Break work into minimal, verifiable steps. Prefer incremental
  changes with clear tests.
- Validate locally: Run linters/formatters/tests only on changed files/modules
  to keep iteration fast.
- Document: Update `README.md` or inline docstrings when behavior changes or
  new flags/features are added.


Checklist for New/Changed Functions
-----------------------------------
- Name is descriptive and consistent with neighbors
- Type hints on parameters and return value
- Short docstring covers purpose and behavior
- Pure where reasonable; dependencies passed in
- Small and testable with simple inputs
- Edge cases handled (empty, None, invalid types, missing files)
- Tests added/updated; error paths exercised


FAQ
---
- How do I configure OpenAI access?
  - Set `OPENAI_API_KEY` in the environment or a local `.env` (see
    `load_client()`).
- How do I preview generated names without transcribing?
  - Use `--list` with `--smart-names` (and optionally `--use-ai`), edit the
    saved cache file, then rerun without `--list`.
- How do I control output filenames?
  - Use `--prefix` parts and/or `--smart-names`. See `--help` in
    `transcribe_video.py` for examples.


Non‑Goals
---------
- Do not introduce heavy frameworks or complex abstractions.
- Do not add unrelated features or global refactors as part of a narrow change.
- Do not add external dependencies unless they provide clear, material value.
