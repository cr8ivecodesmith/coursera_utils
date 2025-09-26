# Markdown to PDF

## **Sources**

Main module:
- `src/study_utils/markdown_to_pdf.py`

## **Description**

Convert one or more Markdown files into a single PDF with configurable paper size, margins, optional table of contents, and an optional title page. Uses WeasyPrint (pure Python HTML→PDF) for portability and simplicity on Linux.

## **Python Compatibility**

- Target: Python 3.8+ (consistent with repo usage). Suggested libs support 3.8–3.12.
- Python libs:
  - `markdown-it-py` (Markdown→HTML)
  - `weasyprint` (HTML→PDF)
  - `jinja2` (title page templating)
  - `pygments` (code highlight CSS)

## **External System Libraries (Linux)**

- WeasyPrint requires Cairo, Pango, and related libs (see README for apt packages).

## **Inputs**

**Required:**
- Output PDF path
- One or more input Markdown files or directories (recursively discovering `*.md` by default)

**Optional:**
- `--paper-size {letter,a4,legal,a5}`: Default `letter`.
- `--orientation {portrait,landscape}`: Default `portrait`.
- `--margin` shorthand (CSS-like, accepts units: `in`, `mm`, `cm`, `pt`), e.g., `--margin 1in` or `--margin "1in 0.75in"`.
  - Edge-specific: `--margin-top`, `--margin-right`, `--margin-bottom`, `--margin-left` (overrides shorthand).
- `--toc/--no-toc`: Include table of contents; default off.
- `--toc-depth N`: Max heading level for TOC; default 3.
- `--title-page`: Include a title page. Use with:
  - `--title`, `--subtitle`, `--author`, `--date` (defaults to today in local timezone)
  - `--title-template PATH` (Jinja2 template)
- `--ai-title`: Generate title page fields with AI (optional). Uses repo’s `load_client()`.
  - `--ai-model MODEL`
  - `--ai-prompt PATH`
  - `--ai-max-tokens`, `--ai-temperature`
- `--css PATH`: Custom stylesheet.
- `--highlight-style {default,monokai,github} | --highlight-css PATH`: Code highlighting.
- `--resources PATH`: Base path for images and assets.
- `--extensions ext1 ext2 ...`: Additional Markdown extensions to pass to parser.
- `--level-limit N`: When inputs include directories, limit recursion depth.
- `--sort {name,created,modified}` with optional `-` prefix for desc.
- `--dry-run`: Print resolved plan without generating output.
- `--verbose`: Print detailed steps.

## **Behavior**

- Discovery: Accept files and/or directories; default extensions `md`, `markdown`. Deterministic sort unless overridden.
- Concatenation: Preserve input order. Assemble a single HTML document from title page (optional), TOC (optional), and Markdown sections converted via `markdown-it-py`.
- Title page:
  - If `--title-page` and fields provided, render via Jinja2 template.
  - If `--ai-title`, call `load_client()` and prompt the model using the first N lines of the first Markdown (configurable) and/or filenames to infer `title`, `subtitle`, `author`.
  - Sanitize AI output; trim to reasonable length; escape HTML.
- Table of contents: Generate from parsed headings up to `--toc-depth`; link with anchors in the assembled HTML.
- Margins, paper, orientation: Apply via CSS `@page` rules; generate stylesheet from options.
- Styles: Ship a default print CSS; allow custom CSS via `--css` to override/extend defaults.
- Output: Write only to the explicit output path. Use a temp build dir under OS temp; clean up on success.
- Errors: Fail fast with precise messages about missing system libs for WeasyPrint or invalid inputs. Offer remediation hints.

## **Configuration Defaults**

- Default paper size: `letter`
- Default margins: `1cm` all sides
- Default orientation: `portrait`
- Default TOC: off
- Default title template: simple centered layout with document metadata; single page
- Optional `study_utils/pdf_defaults.toml` for persistent defaults:
  - `[defaults] paper_size = "a4"; margin = "1cm"; orientation = "portrait"; toc = false
  - `[style] css = "study_utils/resources/print.css"`
  - `[ai] model = "gpt-4o-mini"; max_tokens = 200; temperature = 0.3`

## **CLI Usage**

- `python -m study_utils.markdown_to_pdf OUTPUT.pdf INPUTS... [options]`
- Examples:
- `python -m study_utils.markdown_to_pdf out.pdf notes.md --toc --paper-size a4`
- `python -m study_utils.markdown_to_pdf out.pdf docs/ --level-limit 2 --margin "1in 0.75in"`
- `python -m study_utils.markdown_to_pdf out.pdf intro.md chapter*.md --title-page --title "My Guide" --author "Me"`
- `python -m study_utils.markdown_to_pdf out.pdf README.md --ai-title --toc`

## **Implementation Outline**

- Parse args with `argparse`; keep pure helpers for:
  - Path discovery (`iter_markdown_files`), sorting, and recursion
  - Margin/paper/orientation normalization to CSS `@page` rules
  - Title page rendering with Jinja2
  - Optional AI title generation using `load_client()`; isolated and mockable
  - Markdown→HTML with `markdown-it-py`; assemble single HTML with anchors
  - TOC generation from headings
- Render to PDF using `weasyprint.HTML(string=..., base_url=resources_dir).write_pdf(stylesheets=[...])`.
- Keep I/O isolated in `main()`; core logic pure and unit-testable.

## **Testing**

- Unit tests (no system tools required beyond Python libs):
  - Arg parsing and option normalization (paper/margins/orientation/TOC)
  - Title page templating given explicit fields
  - AI title path with a stubbed client returning deterministic fields
  - TOC generation (given headings produces correct anchors)
  - CSS generation for `@page` with margins and paper size (snapshot)
- Optional integration test marked and skipped if `weasyprint` is not importable

## **Error Messages & Remediation**

- Missing WeasyPrint or system libraries: hint to install Cairo/Pango on Debian/Ubuntu
- Absent `OPENAI_API_KEY` when `--ai-title`: explain `.env` support via `load_client()`

## **Security & Performance**

- No shell interpolation; all operations stay within Python
- Sanitize user paths; disallow writing outside the specified output
- Stream file reads; avoid loading extremely large Markdown into memory at once when concatenating
- Deterministic output given inputs; all randomness/AI optional and explicit
