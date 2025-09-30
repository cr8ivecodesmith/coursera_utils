# Quizzer Rich UI Migration — Spec

## Summary
Replace the Textual-based quizzer interface with a Rich-powered CLI session so `study quizzer start` runs entirely within Rich-rendered prompts, simplifying dependencies while preserving the existing quiz flow and summaries.

## Goals
- Introduce a Rich-driven quiz session module in `study_utils/quizzer/session.py` exposing a first-class `run_quiz_session(questions, console, input_provider, *, show_explanations=True)` entry point that handles question rendering, answer selection, navigation, and submission within a synchronous console loop.
- Return a structured `QuizSessionResult` dataclass from `run_quiz_session` capturing per-question responses, aggregated summary metrics, and the final exit action so callers can assert behaviour without depending on Rich rendering side effects.
- Update the `quizzer start` command to orchestrate sessions via the new Rich UI layer, including shuffle/limit wiring already exposed by CLI flags.
- Retire Textual artifacts (`QuizApp`, `QuestionView`, `quiz.tcss`) along with the `study_utils/quizzer/view` package, and remove the `textual` dependency from `pyproject.toml` and exports in `study_utils.quizzer`.
- Preserve and, if needed, adapt result aggregation helpers so post-quiz summaries (overall accuracy plus per-topic stats) remain available for reuse.
- Refresh unit tests to target the Rich session logic and helpers, ensuring coverage for navigation branches, command parsing, and summary formatting.

## Non-Goals
- No changes to quiz generation, topic extraction, or storage formats for questions/sessions.
- No introduction of asynchronous event loops, mouse support, or advanced terminal features beyond what Rich already offers.
- No implementation of the currently stubbed `review` or `report` commands beyond making sure they still surface the existing placeholder messaging.

## Behavior (BDD-ish)
- Given `study quizzer start <name>` finds a populated `questions.jsonl`, when the user is shown a question, then the Rich-rendered view lists the stem, choices, progress, and available commands (`a`–`d`, `n`, `p`, `submit`, `quit`).
- Given CLI flags such as `--shuffle`, `--num`, and `--explain`, when `quizzer start` invokes the Rich session, then the session API receives those options explicitly (with `show_explanations` driving explanation visibility) while currently unimplemented knobs like `--mix`/`--resume` continue to emit their placeholder messaging.
- Given the user enters a choice key (case-insensitive), when the session accepts the input, then the answer is recorded, the UI updates the selection indicator, and progress counters reflect the total answered questions.
- Given the user enters `submit`, when the session finalizes, then a Rich table summarizing correctness, explanations (when enabled), and aggregated statistics is printed and `run_quiz_session` returns a `QuizSessionResult` containing the rendered responses and summary before control returns to the CLI (which exits with code 0).

## Constraints & Dependencies
- Constraints: stay within synchronous Rich console APIs (no live refresh requirements); maintain compatibility with standard POSIX terminals; avoid regressing automated testability by keeping pure helpers for command parsing and summaries; ensure the new Rich session boundary is the single public UI surface (no leftover Textual exports); keep `input_provider` as a simple `Callable[[], str]` seam so tests and alternate shells can inject scripted input.
- Upstream/Downstream: update CLI entry (`src/study_utils/quizzer/_main.py`), public exports (`src/study_utils/quizzer/__init__.py`), and tests under `tests/test_quiz_session_ui.py` & `tests/test_quizzer_view_extra.py`; ensure documentation that references Textual is revised if present.

## Security & Privacy
- No new data collection; inputs remain local to the terminal. Continue to avoid logging quiz content to unexpected locations and respect existing config-loading behavior.

## Telemetry & Operability
- Provide concise stdout messaging for start/end states and error cases (e.g., empty bank). No additional logging or metrics plumbing is required.

## Rollout / Revert
- Rollout: land the Rich session module, wire CLI changes, update deps/tests, and verify end-to-end via manual CLI smoke test plus automated coverage.
- Revert: restore the prior Textual classes, re-add the dependency, and revert CLI/test changes.

## Definition of Done
- [ ] Behavior verified via automated tests covering session command parsing, navigation, and summary output helpers.
- [ ] Documentation (README/CLI help) updated to reference the Rich-based experience where applicable.
- [ ] Tests added/updated to maintain 100% pytest coverage for touched modules and a clean Ruff run.
- [ ] No lingering Textual artifacts or dependencies in the repository.
- [ ] Manual smoke test instructions captured in implementation log once executed.

## Ownership
- Owner: @matt
- Reviewers: @codex
- Stakeholders: @matt

## Links
- Related code: `src/study_utils/quizzer/_main.py`, `src/study_utils/quizzer/view/quiz.py`, `src/study_utils/quizzer/__init__.py`, `pyproject.toml`, `tests/test_quiz_session_ui.py`, `tests/test_quizzer_view_extra.py`

## History
### 2025-09-30 14:05
**Summary** — Clarified Rich session API contract and flag handling
**Changes**
- Documented `run_quiz_session` signature and CLI flag expectations in goals/behaviour
- Tightened constraints to reflect Rich session as sole UI surface
### 2025-09-30 12:56
**Summary** — Initial Rich migration spec drafted
**Changes**
- Captured scope for replacing Textual quiz UI with Rich, enumerated goals, behaviors, and completion criteria
