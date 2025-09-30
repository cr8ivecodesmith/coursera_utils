# Study Utils

Study Utils is a collection of scripts that support the full study loop for
online courses and self-directed learning.

It captures some of the key ideas of a Personal Knowledge & Learning System (PKLS)
that I've written about on my essays site:
<https://essays.mattlebrun.com/2025/09/personal-knowledge-learning-system.html>

## Features

CLI first tooling for:

- Converting various document formats to Markdown
- Reshaping documents into study-friendly formats
- Create Retrieval-Augmented Generation (RAG) databases for study and topic exploration
- Create and take quizzes to reinforce learning


## Requirements

- Python 3.12+
- `OPENAI_API_KEY` environment variable for AI-powered features (supports `.env`)

Optional but recommended:

- System libraries for WeasyPrint (for Markdown → PDF conversion)
- `ffmpeg` for video transcription (pydub dependency)
- `pandoc` for epub → markdown conversion (if you want epub support)
- `uv` for dependency and virtual environment management
- A markdown document viewer/editor of your choice

This currently tested on Ubuntu 22.04/24.04 (WSL) and Termux on Android.

Technically it should work on macOS, but I haven't tested it there.


## Quick start

Assuming you have the system requirements in place,
you can get started in a few steps.

**Create a study workspace:**

```bash
mkdir -p ~/Study && cd ~/Study
```

**Create configuration files:**

```bash
printf 'OPENAI_API_KEY=your_api_key_here\n' > .env
```

**Install with pip:**

```bash
uv pip install git+https://github.com/cr8ivecodesmith/study-utils.git
```

**Initialize the workspace:**

```bash
uv run study init
```

**Convert documents to Markdown:**

```bash
mkdir -p materials
```

```bash
uv run study convert-markdown materials
```

**Reshape documents to study formats:**

```bash
uv run study generate-document reading_assignment assignments/lesson-01.md ~/.study-utils-data/converted
```

```bash
uv run study generate-document keywords assignments/lesson-01-keywords.md ~/.study-utils-data/converted
```

**Create a RAG database:**

Create a RAG database from the converted materials:

```bash
uv run study rag ingest --name lesson-01 ~/.study-utils-data/converted
```

Explore the study materials:

```bash
uv run study rag chat --db lesson-01
```

**Take quizzes:**

Initialize and edit `quizzer.toml` and set:

```bash
uv run study quizzer init lesson-01
```

```txt
[quiz.lesson-01]
sources = ["/path/to/home/.study-utils-data/converted"]
```

Generate topics and questions:

```bash
uv run study quizzer topics generate lesson-01
```

```bash
uv run study quizzer questions generate lesson-01 --per-topic 5
```

Start a quiz session:

```bash
uv run study quizzer start lesson-01 --num 5
```

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

## Workspace and configuration

- Run `study init` to bootstrap the shared workspace (defaults to
  `~/.study-utils-data`). The command creates `converted/`, `logs/`, and
  `config/` directories and accepts `--path` for alternate locations.
- Configuration files live under `<workspace>/config`. Use
  `study convert-markdown config init` to scaffold `convert_markdown.toml` with
  documented defaults. Pass `--workspace` to target a specific workspace or
  `--path`/`--force` for fully custom destinations.
- All CLI entry points respect the `STUDY_UTILS_DATA_HOME` environment
  variable when resolving the workspace; if unset they fall back to the default
  directory created by `study init`.

## System requirements

### OS Requirements (Ubuntu)

To enable Markdown → PDF generation with WeasyPrint, install its system libraries and fonts.

- Base setup:
  ```bash
  sudo apt-get update
  sudo apt-get install -y libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libxml2 libxslt1.1 shared-mime-info
  ```
- Recommended fonts:
  ```bash
  sudo apt-get install -y fonts-dejavu fonts-liberation
  ```
- To enable video transcription, ensure ffmpeg is installed (pydub uses it under the hood):
  ```
  sudo apt-get install -y ffmpeg
  ```

## Gather materials for a module

- Download all materials for a module.
- At the very least the transcript file for the video.
- If there's no transcript file available, use `study transcribe-video`.

## CLI commands

All tooling is routed through the `study` console script. Run `study list` to
see available commands or `study help <command>` for details and supported
flags.

**`study transcribe-video TARGET [options]`**

- Transcribe one `.mp4` file or a directory of `.mp4` files using Whisper-1.
- Supports optional recursion, list/preview mode, smart names with optional AI
  refinement, and composable filename prefixes.

Examples:

- Preview names and save editable cache: 
  ```bash
  uv run study transcribe-video ./videos --list --smart-names --use-ai
  ```
- Transcribe a directory recursively with smart names and counter prefix:
  ```bash
  study transcribe-video ./videos -r --smart-names -p 'text:Lecture ' -p 'counter:NN'
  ```
- Transcribe a single file to a custom folder: 
  ```bash
  study transcribe-video ./videos/intro.mp4 -o ./transcripts
  ```

**`study markdown-to-pdf OUTPUT.pdf INPUTS... [options]`**

- Convert Markdown to a single PDF using WeasyPrint, with configurable paper
  size, margins, optional title page, and optional table of contents.

Examples:

```bash
study markdown-to-pdf out.pdf notes.md --toc --paper-size a4
```

```bash
study markdown-to-pdf out.pdf docs/ --level-limit 2 --margin "1in 0.75in"
```

```bash
study markdown-to-pdf out.pdf README.md --title-page --title "My Guide" --author "Me"
```

**`study init [options]`**

- Provision the shared workspace directory and any missing subdirectories.

Examples:

```bash
study init
```

```bash
study init --path /tmp/study-utils
```

**`study convert-markdown PATHS... [options]`**

- Convert PDFs, DOCX, HTML, TXT, and EPUB files into Markdown outputs with
  YAML front matter while preserving basenames.
- Default outputs land in `<workspace>/converted`; use `--output-dir` or the
  TOML config to redirect elsewhere.
- Respects layered config precedence (CLI flags > environment variables >
  `convert_markdown.toml`).
- Use `study convert-markdown config init` to scaffold the default template in
  the workspace config directory.

Examples:

```bash
study convert-markdown ./docs --extensions pdf docx
```

```bash
study convert-markdown config init --workspace ~/.study-utils-data
```

**`study text-combiner OUTPUT INPUTS... [options]`**

- Combine text files with optional section titles and ordering. See `--help`.

**`study generate-document DOC_TYPE OUTPUT INPUTS... [options]`**
- Generate a Markdown document from reference files using prompts defined in a
  TOML config.
- Looks for `documents.toml` in the current directory, then falls back to the
  bundled defaults under `study_utils/documents.toml`.

Example: 

```bash
study generate-document reading_assignment notes.md ./materials --extensions txt md --level-limit 0
```

**`study quizzer [options]`**

- Launch the interactive Rich-based quiz session for drilling on generated questions.

**`study rag <subcommand> [options]`**
- Manage retrieval-augmented study databases and chat sessions.
  Includes `config`, `ingest`, `list`, `inspect`, `export`, `import`,
  `chat`, and `doctor` helpers.

Examples:

```bash
study rag config init
```

```bash
study rag ingest --name physics-notes ./notes
```

```bash
study rag doctor
```
