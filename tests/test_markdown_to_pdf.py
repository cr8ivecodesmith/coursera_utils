from pathlib import Path

import pytest

from study_utils.markdown_to_pdf import (
    build_markdown_it,
    build_page_css,
    default_title_template,
    iter_markdown_files,
    parse_margin_shorthand,
    render_markdown_with_headings,
    slugify,
    TitleFields,
    render_title_page,
)


def test_build_page_css_defaults_and_overrides():
    css = build_page_css()
    assert "@page" in css
    assert "Letter portrait" in css
    assert "margin: 1cm 1cm 1cm 1cm" in css

    css2 = build_page_css(paper_size="a4", orientation="landscape", margin_shorthand="1in 0.5in")
    assert "A4 landscape" in css2
    assert "margin: 1in 0.5in 1in 0.5in" in css2

    css3 = build_page_css(margin_shorthand="10mm", margin_left="20mm")
    assert "margin: 10mm 10mm 10mm 20mm" in css3

    with pytest.raises(ValueError):
        build_page_css(paper_size="bogus")
    with pytest.raises(ValueError):
        build_page_css(margin_shorthand="5")  # missing unit


def test_slugify_and_headings_render():
    md = build_markdown_it([])
    used = {}
    text = "# Hello World\n\n## Hello World\n\n## Another section\n"
    html, heads = render_markdown_with_headings(md, text, used)
    # Ensure unique anchors and proper levels
    assert any(h.anchor == "hello-world" and h.level == 1 for h in heads)
    assert any(h.anchor == "hello-world-2" and h.level == 2 for h in heads)
    assert any(h.anchor == "another-section" and h.level == 2 for h in heads)
    assert "id=\"hello-world\"" in html
    assert "id=\"hello-world-2\"" in html


def test_title_page_rendering_default_template():
    fields = TitleFields(title="My Doc", subtitle="An Intro", author="Me", date_str="2025-08-27")
    html = render_title_page(fields)
    assert "My Doc" in html
    assert "An Intro" in html
    assert "Me" in html
    assert "2025-08-27" in html


def test_iter_markdown_files_and_sort(tmp_path: Path):
    d = tmp_path
    a = d / "a.md"; a.write_text("# A", encoding="utf-8")
    b = d / "b.markdown"; b.write_text("# B", encoding="utf-8")
    c = d / "c.txt"; c.write_text("nope", encoding="utf-8")
    sub = d / "sub"; sub.mkdir()
    x = sub / "x.md"; x.write_text("# X", encoding="utf-8")

    files = list(iter_markdown_files([d], level_limit=0))
    # should include a, b, x
    names = sorted([p.name for p in files])
    assert names == ["a.md", "b.markdown", "x.md"]
