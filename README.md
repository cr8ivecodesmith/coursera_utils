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

## Gather materials for a module

- Download all materials for a module.
- At the very least the transcript file for the video.
- If there's no transcript file available, use the `transcribe_video.py` script.

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
