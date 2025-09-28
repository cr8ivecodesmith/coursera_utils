# Study Utils

Study Utils is a collection of scripts that support the full study loop for
online courses and self-directed learning.

## Development setup

- `uv` is the preferred workflow for managing dependencies and virtual
  environments. Run `uv sync --dev` to install everything needed for local
  development.
- When working inside Termux, `pyenv` remains the more reliable option for
  managing Python versions and virtual environments. Install Python 3.12 with
  `pyenv` and create a virtual environment before running the tooling.

## Testing

- Run `uv run pytest` (or `just test`) to execute the full suite. Coverage is
  enforced at 100% via `pytest-cov`; any regression will fail the run.
- The tests run entirely offline by default. Shared fixtures under
  `tests/fixtures/` provide stubs for OpenAI, WeasyPrint, dotenv, and pydub to
  keep runs deterministic.
- For local debugging you can generate a detailed report with
  `uv run pytest --cov-report=term-missing`. To temporarily relax the coverage
  gate while debugging, drop `--cov-fail-under=100` from `pyproject.toml` and
  restore it before committing.

## System requirements

### OS Requirements (Ubuntu)

To enable Markdown → PDF generation with WeasyPrint, install its system libraries and fonts.

- Base setup:
  - `sudo apt-get update`
  - `sudo apt-get install -y libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libxml2 libxslt1.1 shared-mime-info`
  - Recommended fonts: `sudo apt-get install -y fonts-dejavu fonts-liberation`

To enable video transcription, ensure ffmpeg is installed (pydub uses it under the hood):

- `sudo apt-get install -y ffmpeg`

Notes:
- The PDF generator uses WeasyPrint only; Pandoc/LaTeX is not required.
- Transcription and document generation that leverage AI require `OPENAI_API_KEY` (supports `.env`).

## Gather materials for a module

- Download all materials for a module.
- At the very least the transcript file for the video.
- If there's no transcript file available, use `study transcribe-video`.

## CLI commands

All tooling is routed through the `study` console script. Run `study list` to
see available commands or `study help <command>` for details and supported
flags.

- `study transcribe-video TARGET [options]`
  - Transcribe one `.mp4` file or a directory of `.mp4` files using Whisper-1.
  - Supports optional recursion, list/preview mode, smart names with optional AI
    refinement, and composable filename prefixes.
  - Examples:
    - Preview names and save editable cache: `study transcribe-video ./videos --list --smart-names --use-ai`
    - Transcribe a directory recursively with smart names and counter prefix:
      `study transcribe-video ./videos -r --smart-names -p 'text:Lecture ' -p 'counter:NN'`
    - Transcribe a single file to a custom folder: `study transcribe-video ./videos/intro.mp4 -o ./transcripts`
- `study markdown-to-pdf OUTPUT.pdf INPUTS... [options]`
  - Convert Markdown to a single PDF using WeasyPrint, with configurable paper
    size, margins, optional title page, and optional table of contents.
  - Examples:
    - `study markdown-to-pdf out.pdf notes.md --toc --paper-size a4`
    - `study markdown-to-pdf out.pdf docs/ --level-limit 2 --margin "1in 0.75in"`
    - `study markdown-to-pdf out.pdf README.md --title-page --title "My Guide" --author "Me"`
- `study convert-markdown PATHS... [options]`
  - Convert PDFs, DOCX, HTML, TXT, and EPUB files into Markdown outputs with
    YAML front matter while preserving basenames.
  - Use `study convert-markdown config init` to scaffold the default
    `convert_markdown.toml` in the workspace config directory.
  - Examples:
    - `study convert-markdown ./docs --extensions pdf docx`
    - `study convert-markdown config init --workspace ~/.study-utils-data`
- `study text-combiner OUTPUT INPUTS... [options]`
  - Combine text files with optional section titles and ordering. See `--help`.
- `study generate-document DOC_TYPE OUTPUT INPUTS... [options]`
  - Generate a Markdown document from reference files using prompts defined in a
    TOML config.
  - Looks for `documents.toml` in the current directory, then falls back to the
    bundled defaults under `study_utils/documents.toml`.
  - Example: `study generate-document reading_assignment notes.md ./materials --extensions txt md --level-limit 0`
- `study quizzer [options]`
  - Launch the interactive quizzer TUI for drilling on generated questions.
- `study rag <subcommand> [options]`
  - Manage retrieval-augmented study databases and chat sessions.
    Includes `config`, `ingest`, `list`, `inspect`, `export`, `import`,
    `chat`, and `doctor` helpers.
  - Examples:
    - `study rag config init`
    - `study rag ingest --name physics-notes ./notes`
    - `study rag doctor`
