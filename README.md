Coursera Utils
===

A collection of scripts to help with studying.

The main process is to:

1. Gather materials for a module
2. Pre-study
3. Take the online quizzes / exercise
4. Deep dive on missed questions
5. Repeat steps 3-4 until perfect
7. Repeat step 1 for the next module

In each step you should be taking non-linear notes focusing on connecting
concepts with one another and even more importantly, your own experiences.

Each step will have substeps helped by the scripts in this repository.

## Requirements

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
- If there's no transcript file available, use `python -m app.transcribe_video`.

## Pre-study

## Take the online quizzes / exercise

## Deep dive on missed questions

## Repeat steps 3-4 until perfect

## Write summary notes of most relevant concepts and how to use them

## Repeat step 1 for the next module

## Utilities

- `python -m app.text_combiner OUTPUT INPUTS... [options]`
  - Combine text files with optional section titles and ordering. See `--help`.
- `python -m app.generate_document DOC_TYPE OUTPUT INPUTS... [options]`
  - Generate a Markdown document from reference files using prompts defined in a TOML config.
  - Looks for `documents.toml` in the current directory, then falls back to the bundled defaults under `app/documents.toml`.
  - Example: `python -m app.generate_document reading_assignment notes.md ./materials --extensions txt md --level-limit 0`
- `python -m app.transcribe_video TARGET [options]`
  - Transcribe one `.mp4` file or a directory of `.mp4` files using Whisper-1.
  - Supports optional recursion, list/preview mode, smart names with optional AI refinement, and composable filename prefixes.
  - Examples:
    - Preview names and save editable cache: `python -m app.transcribe_video ./videos --list --smart-names --use-ai`
    - Transcribe a directory recursively with smart names and counter prefix: `python -m app.transcribe_video ./videos -r --smart-names -p 'text:Lecture ' -p 'counter:NN'`
    - Transcribe a single file to a custom folder: `python -m app.transcribe_video ./videos/intro.mp4 -o ./transcripts`
- `python -m app.markdown_to_pdf OUTPUT.pdf INPUTS... [options]`
  - Convert Markdown to a single PDF using WeasyPrint, with configurable paper size, margins, optional title page, and optional table of contents.
  - Examples:
    - `python -m app.markdown_to_pdf out.pdf notes.md --toc --paper-size a4`
    - `python -m app.markdown_to_pdf out.pdf docs/ --level-limit 2 --margin "1in 0.75in"`
    - `python -m app.markdown_to_pdf out.pdf README.md --title-page --title "My Guide" --author "Me"`
