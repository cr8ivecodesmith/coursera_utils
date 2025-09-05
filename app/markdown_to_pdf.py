"""Markdown to PDF generator (WeasyPrint-only backend).

Features:
- Discover and concatenate Markdown files to a single PDF.
- Configurable paper size, orientation, and margins via CSS @page.
- Optional table of contents generated from headings.
- Optional title page via Jinja2; optional AI-generated fields using load_client().

Design:
- Pure helper functions for discovery, CSS generation, Markdown→HTML, TOC, and templating.
- I/O and WeasyPrint usage isolated in main().
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from jinja2 import Environment, FileSystemLoader, Template
from markdown_it import MarkdownIt
from pygments.formatters import HtmlFormatter


# ------------- Types and constants -------------

PAPER_SIZES = {"letter": "Letter", "a4": "A4", "legal": "Legal", "a5": "A5"}


@dataclass
class Margin:
    top: str
    right: str
    bottom: str
    left: str


@dataclass
class TitleFields:
    title: str = ""
    subtitle: str = ""
    author: str = ""
    date_str: str = ""


# ------------- Discovery -------------

def _is_md(path: Path, exts: Sequence[str]) -> bool:
    e = path.suffix.lower().lstrip(".")
    return path.is_file() and e in {x.lower().lstrip(".") for x in exts}


def iter_markdown_files(
    inputs: Sequence[Path],
    *,
    extensions: Sequence[str] = ("md", "markdown"),
    level_limit: int = 0,
) -> Iterator[Path]:
    """Yield markdown files from the given file/dir inputs.

    - level_limit=0 means unlimited recursion; >=1 limits depth relative to each starting dir.
    """
    for p in inputs:
        p = p.expanduser().resolve()
        if p.is_file() and _is_md(p, extensions):
            yield p
        elif p.is_dir():
            base_depth = len(p.parts)
            for child in sorted(p.rglob("*")):
                if not child.is_file():
                    continue
                if level_limit and len(child.parts) - base_depth > level_limit:
                    continue
                if _is_md(child, extensions):
                    yield child


def sort_files(files: Sequence[Path], key: str = "name") -> List[Path]:
    """Return a sorted list by the requested key.

    key: name|created|modified with optional '-' prefix for descending.
    """
    desc = False
    k = key
    if key.startswith("-"):
        desc = True
        k = key[1:]

    def _key(p: Path):
        if k == "name":
            return p.name.lower()
        if k == "created":
            try:
                return p.stat().st_ctime
            except Exception:
                return 0
        if k == "modified":
            try:
                return p.stat().st_mtime
            except Exception:
                return 0
        # default fallback
        return p.name.lower()

    return sorted(files, key=_key, reverse=desc)


# ------------- CSS generation -------------

_CSS_UNIT_RE = re.compile(r"^(?:\d+\.?\d*|\d*\.\d+)(?:in|cm|mm|pt)$")


def _validate_unit(value: str) -> str:
    v = value.strip()
    if not _CSS_UNIT_RE.match(v):
        raise ValueError(
            f"Invalid CSS size '{value}'. Use units in, mm, cm, pt (e.g., '1in', '10mm')."
        )
    return v


def parse_margin_shorthand(margin: Optional[str]) -> Optional[Margin]:
    if not margin:
        return None
    parts = [p for p in margin.strip().split() if p]
    vals = [_validate_unit(p) for p in parts]
    if len(vals) == 1:
        t = r = b = l = vals[0]  #noqa - fix later
    elif len(vals) == 2:
        t = b = vals[0]
        r = l = vals[1]  #noqa - fix later
    elif len(vals) == 3:
        t, r, b = vals
        l = r  #noqa - fix later
    elif len(vals) == 4:
        t, r, b, l = vals  #noqa - fix later
    else:
        raise ValueError("Margin accepts 1-4 CSS size values (e.g., '1in' or '1in 0.5in').")
    return Margin(top=t, right=r, bottom=b, left=l)


def build_page_css(
    *,
    paper_size: str = "letter",
    orientation: str = "portrait",
    margin_shorthand: Optional[str] = None,
    margin_top: Optional[str] = None,
    margin_right: Optional[str] = None,
    margin_bottom: Optional[str] = None,
    margin_left: Optional[str] = None,
) -> str:
    size_keyword = PAPER_SIZES.get(paper_size.lower())
    if not size_keyword:
        raise ValueError(f"Unsupported paper size: {paper_size}. Choose from {sorted(PAPER_SIZES)}")
    if orientation not in {"portrait", "landscape"}:
        raise ValueError("orientation must be 'portrait' or 'landscape'")

    base = parse_margin_shorthand(margin_shorthand) or Margin("1cm", "1cm", "1cm", "1cm")
    top = _validate_unit(margin_top) if margin_top else base.top
    right = _validate_unit(margin_right) if margin_right else base.right
    bottom = _validate_unit(margin_bottom) if margin_bottom else base.bottom
    left = _validate_unit(margin_left) if margin_left else base.left

    return (
        "@page {\n"
        f"  size: {size_keyword} {orientation};\n"
        f"  margin: {top} {right} {bottom} {left};\n"
        "}\n"
    )


def default_highlight_css(style_name: str = "default") -> str:
    formatter = HtmlFormatter(style=style_name)
    return formatter.get_style_defs(".highlight")


# ------------- Markdown rendering and TOC -------------

SLUG_CHARS_RE = re.compile(r"[^a-z0-9\- ]+")
WHITESPACE_RE = re.compile(r"\s+")


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = SLUG_CHARS_RE.sub("", s)
    s = WHITESPACE_RE.sub("-", s).strip("-")
    return s or "section"


@dataclass
class Heading:
    level: int
    text: str
    anchor: str


def render_markdown_with_headings(md: MarkdownIt, text: str, used: Dict[str, int]) -> Tuple[str, List[Heading]]:
    """Render markdown to HTML while injecting unique heading anchors and returning headings.

    Modifies token tree to add id attributes for heading_open tokens.
    """
    tokens = md.parse(text)
    headings: List[Heading] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == "heading_open" and i + 1 < len(tokens) and tokens[i + 1].type == "inline":
            level = int(t.tag[1:]) if t.tag.startswith("h") else 1
            content = tokens[i + 1].content
            base = slugify(content)
            n = used.get(base, 0)
            anchor = base if n == 0 else f"{base}-{n+1}"
            used[base] = n + 1
            # attach id attr robustly across markdown-it-py versions
            try:
                # Prefer official API
                t.attrSet("id", anchor)  # type: ignore[attr-defined]
            except Exception:
                # Fallback: handle attrs as dict or list
                if t.attrs is None:
                    try:
                        t.attrs = {"id": anchor}  # type: ignore[assignment]
                    except Exception:
                        t.attrs = [["id", anchor]]  # type: ignore[assignment]
                else:
                    if isinstance(t.attrs, dict):  # type: ignore[redundant-cast]
                        t.attrs["id"] = anchor
                    else:
                        # assume list of [name, value]
                        found = False
                        for pair in t.attrs:
                            if pair[0] == "id":
                                pair[1] = anchor
                                found = True
                                break
                        if not found:
                            t.attrs.append(["id", anchor])
            headings.append(Heading(level=level, text=content, anchor=anchor))
            i += 2
            continue
        i += 1
    html = md.renderer.render(tokens, md.options, {})
    return html, headings


def build_toc_html(headings: Sequence[Heading], max_depth: int = 3) -> str:
    if max_depth < 1:
        return ""
    # Simple nested list by heading levels up to max_depth
    items: List[str] = []
    stack: List[int] = []
    prev_level = 0
    for h in headings:
        if h.level > max_depth:
            continue
        level = h.level
        if prev_level == 0:
            items.append("<ul class=\"toc\">")
        elif level > prev_level:
            items.append("<ul>")
        elif level < prev_level:
            for _ in range(prev_level - level):
                items.append("</li></ul>")
        else:
            items.append("</li>")
        items.append(f"<li><a href=\"#{h.anchor}\">{escape(h.text)}</a>")
        prev_level = level
    # close remaining lists
    if prev_level:
        items.append("</li>")
        while prev_level > 1:
            items.append("</ul></li>")
            prev_level -= 1
        items.append("</ul>")
    return "".join(items)


def default_title_template() -> Template:
    tpl = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body { font-family: sans-serif; }
        .title-page { display: flex; height: 100vh; align-items: center; justify-content: center; }
        .box { text-align: center; }
        h1 { font-size: 40pt; margin: 0 0 0.4em; }
        h2 { font-size: 20pt; margin: 0 0 1.2em; color: #555; }
        .meta { color: #666; font-size: 11pt; }
      </style>
    </head>
    <body>
      <section class="title-page">
        <div class="box">
          {% if title %}<h1>{{ title }}</h1>{% endif %}
          {% if subtitle %}<h2>{{ subtitle }}</h2>{% endif %}
          <div class="meta">
            {% if author %}<div>By {{ author }}</div>{% endif %}
            {% if date_str %}<div>{{ date_str }}</div>{% endif %}
          </div>
        </div>
      </section>
      <div style="page-break-after: always;"></div>
    </body>
    </html>
    """
    env = Environment(autoescape=True)
    return env.from_string(tpl)


def render_title_page(fields: TitleFields, template_path: Optional[Path] = None) -> str:
    if template_path:
        env = Environment(loader=FileSystemLoader(str(template_path.parent)), autoescape=True)
        tpl = env.get_template(template_path.name)
    else:
        tpl = default_title_template()
    return tpl.render(title=fields.title, subtitle=fields.subtitle, author=fields.author, date_str=fields.date_str)


# ------------- AI title -------------

def generate_ai_title_fields(
    *,
    sample_text: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 200,
    temperature: float = 0.3,
) -> TitleFields:
    """Use repo's load_client to ask the model for title/subtitle/author.

    Returns empty fields on any error, to keep behavior predictable.
    """
    try:
        try:
            # Local import to avoid hard dependency on import time
            from .transcribe_video import load_client  # type: ignore
        except Exception:
            from app.transcribe_video import load_client  # type: ignore
    except Exception:
        return TitleFields()

    try:
        client = load_client()
    except Exception:
        return TitleFields()

    system = "You write concise document metadata as JSON."
    user = (
        "Given the following sample content of a document, propose a short title, optional subtitle, and author.\n"
        "Respond as JSON with keys: title, subtitle, author. Keep title <= 80 chars.\n\n"
        f"Content sample:\n{sample_text[:2000]}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = resp.choices[0].message.content.strip()
        # Try to parse JSON-ish content
        import json

        data = {}
        try:
            data = json.loads(raw)
        except Exception:
            # naive extraction
            m = re.search(r"title\s*[:=]\s*\"([^\"]+)\"", raw, re.I)
            if m:
                data["title"] = m.group(1)
        return TitleFields(
            title=str(data.get("title") or "").strip(),
            subtitle=str(data.get("subtitle") or "").strip(),
            author=str(data.get("author") or "").strip(),
            date_str=date.today().isoformat(),
        )
    except Exception:
        return TitleFields()


# ------------- Assembly and WeasyPrint output -------------

def load_default_css_path() -> Optional[Path]:
    # default CSS next to this module under resources/print.css
    here = Path(__file__).resolve().parent
    css_path = here / "resources" / "print.css"
    return css_path if css_path.exists() else None


def assemble_html(
    parts: Sequence[Tuple[str, str]],
    *,
    include_toc: bool,
    toc_depth: int,
    highlight_css: str,
    custom_css_href: Optional[str] = None,
    title_html: Optional[str] = None,
) -> str:
    """Build a full HTML document string from rendered parts.

    parts: list of (section_title, html_content)
    """
    # Base styles: reset-ish plus pygments plus optional custom.
    base_css = [
        "body { font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; color: #111; line-height: 1.4; }",
        "h1,h2,h3,h4,h5,h6 { page-break-after: avoid; }",
        ".toc { font-size: 0.95em; } .toc a { text-decoration: none; color: inherit; }",
        "pre, code { font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace; }",
        ".highlight { background: #f7f7f7; padding: 0.6em; overflow-x: auto; }",
    ]
    base_css.append(highlight_css)

    default_css_path = load_default_css_path()
    default_css_link = (
        f"<link rel=\"stylesheet\" href=\"{default_css_path.as_posix()}\">" if default_css_path else ""
    )
    custom_css_link = f"<link rel=\"stylesheet\" href=\"{escape(custom_css_href)}\">" if custom_css_href else ""

    # Construct body with optional title and TOC
    body_parts: List[str] = []
    if title_html:
        body_parts.append(title_html)

    # Build combined headings for TOC
    if include_toc:
        # naive TOC: expect each part contains its own headings; combine top-level
        # In practice, we collected headings during rendering and could pass them separately if needed.
        # For simplicity, users get a section list.
        toc_items = [f"<li>{escape(title)}</li>" for title, _ in parts if title]
        if toc_items:
            body_parts.append("<nav class=\"toc-root\"><h2>Table of Contents</h2><ul>" + "".join(toc_items) + "</ul></nav>")
            body_parts.append("<div style=\"page-break-after: always;\"></div>")

    for idx, (title, html) in enumerate(parts):
        if title:
            body_parts.append(f"<h1>{escape(title)}</h1>")
        body_parts.append(html)
        if idx < len(parts) - 1:
            body_parts.append("<div style=\"page-break-after: always;\"></div>")

    doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
      {"\n".join(base_css)}
      </style>
      {default_css_link}
      {custom_css_link}
    </head>
    <body>
      {"".join(body_parts)}
    </body>
    </html>
    """
    return doc


def build_markdown_it(extensions: Sequence[str]) -> MarkdownIt:
    md = MarkdownIt("commonmark", options_update={"html": True})
    # Enable a few common extras if requested
    for ext in extensions:
        e = ext.strip().lower()
        try:
            md.enable(e)
        except Exception:
            # silently ignore unknown rules to keep CLI robust
            pass
    return md


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Convert Markdown files to a single PDF (WeasyPrint)")
    p.add_argument("OUTPUT", help="Output PDF path")
    p.add_argument("INPUTS", nargs="+", help="Markdown files and/or directories")
    p.add_argument("--paper-size", choices=sorted(PAPER_SIZES.keys()), default="letter")
    p.add_argument("--orientation", choices=["portrait", "landscape"], default="portrait")
    p.add_argument("--margin", dest="margin", help="CSS margin shorthand (e.g., '1in' or '1in 0.5in')")
    p.add_argument("--margin-top")
    p.add_argument("--margin-right")
    p.add_argument("--margin-bottom")
    p.add_argument("--margin-left")
    p.add_argument("--toc", dest="toc", action="store_true")
    p.add_argument("--no-toc", dest="toc", action="store_false")
    p.set_defaults(toc=False)
    p.add_argument("--toc-depth", type=int, default=3)
    p.add_argument("--title-page", action="store_true")
    p.add_argument("--title")
    p.add_argument("--subtitle")
    p.add_argument("--author")
    p.add_argument("--date")
    p.add_argument("--title-template", dest="title_template")
    p.add_argument("--ai-title", dest="ai_title", action="store_true")
    p.add_argument("--ai-model", default="gpt-4o-mini")
    p.add_argument("--ai-max-tokens", type=int, default=200)
    p.add_argument("--ai-temperature", type=float, default=0.3)
    p.add_argument("--css", dest="css")
    p.add_argument("--highlight-style", default="default")
    p.add_argument("--resources", help="Base path for images/assets")
    p.add_argument("--extensions", nargs="*", default=[])
    p.add_argument("--level-limit", type=int, default=0)
    p.add_argument("--sort", default="name", help="Sort by name|created|modified with optional '-' for desc")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)

    out_path = Path(args.OUTPUT).expanduser().resolve()
    input_paths = [Path(s) for s in args.INPUTS]
    files = list(iter_markdown_files(input_paths, extensions=("md", "markdown"), level_limit=args.level_limit))
    files = sort_files(files, args.sort)
    if not files:
        raise SystemExit("No markdown files found in inputs")

    # Build CSS strings
    try:
        page_css = build_page_css(
            paper_size=args.paper_size,
            orientation=args.orientation,
            margin_shorthand=args.margin,
            margin_top=args.margin_top,
            margin_right=args.margin_right,
            margin_bottom=args.margin_bottom,
            margin_left=args.margin_left,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    highlight_css = default_highlight_css(args.highlight_style)

    # Prepare Markdown renderer
    md = build_markdown_it(args.extensions)

    # Render each file; collect headings per file for potential TOC
    used_slugs: Dict[str, int] = {}
    parts: List[Tuple[str, str]] = []
    # Also capture a sample of text for optional AI title
    sample_buf: List[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        sample_buf.append(text[:500])
        html, _heads = render_markdown_with_headings(md, text, used_slugs)
        parts.append((f.stem, html))

    # Title page
    title_html: Optional[str] = None
    if args.title_page:
        fields = TitleFields(
            title=args.title or "",
            subtitle=args.subtitle or "",
            author=args.author or "",
            date_str=(args.date or date.today().isoformat()),
        )
        if args.ai_title:
            ai_fields = generate_ai_title_fields(
                sample_text="\n\n".join(sample_buf),
                model=args.ai_model,
                max_tokens=args.ai_max_tokens,
                temperature=args.ai_temperature,
            )
            # only fill missing fields
            fields = TitleFields(
                title=fields.title or ai_fields.title,
                subtitle=fields.subtitle or ai_fields.subtitle,
                author=fields.author or ai_fields.author,
                date_str=fields.date_str or ai_fields.date_str,
            )
        tpl_path = Path(args.title_template).expanduser().resolve() if args.title_template else None
        title_html = render_title_page(fields, tpl_path)

    # Dry run output
    if args.dry_run:
        print("Planned output:")
        print(f"- Output: {out_path}")
        print(f"- Files ({len(files)}):")
        for f in files:
            print(f"  - {f}")
        print(f"- Paper: {args.paper_size} {args.orientation}")
        print(f"- TOC: {'on' if args.toc else 'off'} (depth {args.toc_depth})")
        print(f"- Title page: {'on' if args.title_page else 'off'}{' + AI' if args.title_page and args.ai_title else ''}")
        return

    # Build HTML doc
    html_doc = assemble_html(
        parts,
        include_toc=args.toc,
        toc_depth=args.toc_depth,
        highlight_css=highlight_css,
        custom_css_href=args.css,
        title_html=title_html,
    )

    # Render PDF using WeasyPrint
    try:
        from weasyprint import HTML, CSS
    except Exception as exc:
        raise SystemExit(
            "WeasyPrint is required. Install system libraries (Cairo, Pango) and the 'weasyprint' package."
        )

    base_url = Path(args.resources).expanduser().resolve().as_uri() if args.resources else Path.cwd().as_uri()
    stylesheets = [CSS(string=page_css)]
    if args.css:
        stylesheets.append(CSS(filename=str(Path(args.css).expanduser().resolve())))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_doc, base_url=base_url).write_pdf(target=str(out_path), stylesheets=stylesheets)
    if args.verbose:
        print(f"Wrote PDF to {out_path}")


if __name__ == "__main__":
    main()
