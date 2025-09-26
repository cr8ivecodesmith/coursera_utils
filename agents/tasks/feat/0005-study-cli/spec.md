# Study CLI Wrapper

## **Sources**

New module:
- `src/study_utils/cli.py`

Existing modules leveraged:
- `src/study_utils/transcribe_video.py`
- `src/study_utils/markdown_to_pdf.py`
- `src/study_utils/text_combiner.py`
- `src/study_utils/generate_document.py`
- `src/study_utils/quizzer/_main.py`

Packaging & docs:
- `pyproject.toml` (`[project.scripts]` entries)
- `README.md`
- `agents/tasks/feat/0002-markdown-to-pdf/spec.md` (for context on tool behavior)

## **Description**

Provide a single `study` command that acts as an umbrella for the existing study utilities so people run
`study <tool>` instead of memorizing multiple console scripts. The command should surface available
subcommands, forward arguments to the underlying tool modules, and make it easy to discover what each tool
does. While doing this, add the missing console entry point for the Markdown-to-PDF tool so it is directly
invokable and available through the new wrapper.

## **Goals**

- Ship a `study` console entry point that owns discovery/help and hands off execution to the existing
  tool entry points.
- Ensure every existing CLI (transcribe-video, text-cominer, generate-document, quizzer, markdown-to-pdf)
  is reachable via `study <command> [...]` with no behavior regressions.
- Remove the standalone console scripts (`transcribe-video`, `text-combiner`, etc.) in favor of the unified
  `study` command, even if that breaks older workflows or tests.
- Improve command discoverability with built-in listing and per-command help summaries.
- Keep the implementation easily testable (pure helpers, dependency injection, no implicit `sys.exit`).

## **CLI Behavior**

- Invocation: `study <command> [args...]`.
- With no arguments (or with `study help` / `study --help`), display a usage banner plus an aligned list/table of
  available commands, each with a short, one-line description and a hint to run `study help <command>` for
  details.
- `study list`: Prints the same command table without exiting with an error.
- `study help <command>` (alias `study <command> --help` when feasible):
  - If the underlying module exposes a help renderer (e.g., argparse), invoke it so the user sees the same
    help they would get from running the script directly.
  - On unknown commands, print friendly error plus the list of valid commands.
- Subcommand execution (`study transcribe-video …` etc.):
  - Dispatch to the corresponding module `main(argv)` function so behavior stays identical.
  - Capture and proxy exit codes—if a subcommand raises `SystemExit(code)`, exit the wrapper with the same code.
  - Preserve stdout/stderr streams from the subcommand; do not swallow progress output.
- Built-ins:
  - `study version`: Print the package version (same value as `study-utils --version` in future).
  - `study --version` / `study -V`: same as above.
- Allow `--` to pass through literal arguments (e.g., `study text-combiner -- --help` if the built-in help alias
  conflicts).

## **Command Registry**

- Maintain a small data structure (e.g., `COMMANDS: dict[str, CommandSpec]`) describing each subcommand:
  - `name`, `summary`, callable to invoke (module `main`).
  - Optional flag to mark commands as TUI/long-running (quizzer) so the wrapper can hint accordingly.
- Do not provide command aliases; stick to the canonical command names.
- Prefer lazy imports (or import on registration) to avoid unnecessary startup cost, but keep the code simple.
  Document decisions inside the module with short comments.

## **Pyproject Updates**

- Replace the existing per-tool `[project.scripts]` entries with:
  - `study = "study_utils.cli:main"`
  - `markdown-to-pdf = "study_utils.markdown_to_pdf:main"` (new entry point invoked through the wrapper).
- Remove legacy script entries like `transcribe-video`, `text-combiner`, etc., since we are not maintaining
  backwards compatibility.

## **Documentation**

- Update `README.md` (CLI usage section) to promote `study` as the primary entry point, with examples:
  - `study list`
  - `study transcribe-video <args>`
  - `study markdown-to-pdf <args>`
- Mention that the previous standalone commands have been consolidated into `study`.

## **Testing**

- Unit tests for `study_utils.cli` covering:
  - No-arg invocation and `study help`/`study --help` print usage and return a non-zero exit code appropriate for
    help (likely 2 when omitting a subcommand, 0 when explicitly asking for help).
  - `study list` and `study help <cmd>` produce the expected summary strings.
  - Subcommand dispatch: patch/spy each module `main` to assert it is called with the forwarded argv.
  - Unknown command handling returns a non-zero exit code and includes the suggestion list.
  - `study --version` matches the package version from `importlib.metadata.version`.
- Mark tests that would launch the quizzer TUI to ensure they do not run the actual interface (use mocking).
- Optional golden test to snapshot the command table to catch accidental regressions in wording/order.

## **Non-Goals / Notes**

- Do not redesign the existing tool CLIs; focus on orchestration. Any inconsistencies in their interfaces
  remain out of scope unless they block the wrapper.
- Autocompletion/shell integration is nice-to-have but not required for this iteration.
- Keep the surface ASCII-friendly and avoid colorized output to keep tests simple.

