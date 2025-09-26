from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import study_utils.markdown_to_pdf as mdp
from study_utils.markdown_to_pdf import (
    TitleFields,
    build_markdown_it,
    build_page_css,
    iter_markdown_files,
    render_markdown_with_headings,
    render_title_page,
)


def test_build_page_css_defaults_and_overrides() -> None:
    css = build_page_css()
    assert "@page" in css and "Letter portrait" in css
    assert "margin: 1cm 1cm 1cm 1cm" in css

    css2 = build_page_css(
        paper_size="a4", orientation="landscape", margin_shorthand="1in 0.5in"
    )
    assert "A4 landscape" in css2
    assert "margin: 1in 0.5in 1in 0.5in" in css2

    css3 = build_page_css(margin_shorthand="10mm", margin_left="20mm")
    assert "margin: 10mm 10mm 10mm 20mm" in css3

    with pytest.raises(ValueError):
        build_page_css(paper_size="bogus")
    with pytest.raises(ValueError):
        build_page_css(margin_shorthand="5")
    with pytest.raises(ValueError):
        build_page_css(orientation="diagonal")


@pytest.mark.parametrize(
    "value",
    ["1in", "1in 2in", "1in 2in 3in", "1in 2in 3in 4in"],
)
def test_parse_margin_shorthand_variations(value: str) -> None:
    margin = mdp.parse_margin_shorthand(value)
    assert margin is not None
    assert margin.top


def test_parse_margin_shorthand_invalid_count() -> None:
    with pytest.raises(ValueError):
        mdp.parse_margin_shorthand("1in 2in 3in 4in 5in")


def test_slugify_and_headings_render() -> None:
    md = build_markdown_it([])
    used = {}
    text = "# Hello World\n\n## Hello World\n\n## Another section\n"
    html, heads = render_markdown_with_headings(md, text, used)
    anchors = {h.anchor for h in heads}
    assert "hello-world" in anchors
    assert "hello-world-2" in anchors
    assert "another-section" in anchors
    assert 'id="hello-world"' in html
    assert 'id="hello-world-2"' in html


def test_build_toc_html_nested() -> None:
    headings = [
        mdp.Heading(level=1, text="Intro", anchor="intro"),
        mdp.Heading(level=2, text="Part", anchor="part"),
        mdp.Heading(level=1, text="End", anchor="end"),
    ]
    toc = mdp.build_toc_html(headings, max_depth=2)
    assert "Intro" in toc and "Part" in toc and "End" in toc
    assert toc.startswith('<ul class="toc">')


def test_title_page_rendering_default_template() -> None:
    fields = TitleFields(
        title="My Doc", subtitle="An Intro", author="Me", date_str="2025-08-27"
    )
    html = render_title_page(fields)
    assert "My Doc" in html and "An Intro" in html and "2025-08-27" in html


def test_render_title_page_with_template(tmp_path: Path) -> None:
    template_dir = tmp_path / "tpl"
    template_dir.mkdir()
    template_path = template_dir / "title.html"
    template_path.write_text("Hello {{ title }}", encoding="utf-8")

    fields = TitleFields(title="Custom")
    html = render_title_page(fields, template_path)
    assert "Custom" in html


def test_iter_markdown_files_and_sort(tmp_path: Path) -> None:
    d = tmp_path
    a = d / "a.md"
    a.write_text("# A", encoding="utf-8")
    b = d / "b.markdown"
    b.write_text("# B", encoding="utf-8")
    c = d / "c.txt"
    c.write_text("nope", encoding="utf-8")
    sub = d / "sub"
    sub.mkdir()
    x = sub / "x.md"
    x.write_text("# X", encoding="utf-8")

    files = list(iter_markdown_files([d], level_limit=0))
    names = sorted([p.name for p in files])
    assert names == ["a.md", "b.markdown", "x.md"]

    sorted_desc = mdp.sort_files(files, "-name")
    assert sorted_desc[0].name == "x.md"

    fallback = mdp.sort_files(files, "size")
    assert set(fallback) == set(files)


def test_sort_files_created_and_modified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")

    stats = {
        a: SimpleNamespace(st_ctime=2, st_mtime=2),
        b: SimpleNamespace(st_ctime=1, st_mtime=1),
    }

    original_stat = Path.stat

    def fake_stat(self: Path, *args, **kwargs):
        if self == a:
            raise OSError("boom")
        if self in stats:
            return stats[self]
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    ordered_created = mdp.sort_files([a, b], "created")
    assert ordered_created == [a, b]

    ordered_modified = mdp.sort_files([a, b], "modified")
    assert ordered_modified == [a, b]


def test_render_markdown_with_headings_attr_fallback() -> None:
    class FakeToken:
        def __init__(
            self,
            token_type: str,
            tag: str = "",
            content: str = "",
            attrs=None,
            mode: str = "normal",
        ):
            self.type = token_type
            self.tag = tag
            self.content = content
            self._attrs = attrs
            self.mode = mode
            self._fail_handled = False

        def attrSet(self, *_args, **_kwargs):  # noqa: N802 - matches markdown-it API
            raise RuntimeError("attrSet not available")

        @property
        def attrs(self):
            return self._attrs

        @attrs.setter
        def attrs(self, value):
            if self.mode == "raise_on_set" and not self._fail_handled:
                self._fail_handled = True
                raise RuntimeError("no assign")
            self._attrs = value

    tokens = [
        FakeToken("heading_open", "h1", attrs=None, mode="raise_on_set"),
        FakeToken("inline", content="First"),
        FakeToken("heading_open", "h2", attrs={}),
        FakeToken("inline", content="Second"),
        FakeToken("heading_open", "h3", attrs=[["id", "old"], ["class", "x"]]),
        FakeToken("inline", content="Third"),
        FakeToken("heading_open", "h4", attrs=[["class", "x"]]),
        FakeToken("inline", content="Fourth"),
    ]

    class FakeRenderer:
        @staticmethod
        def render(tokens, _options, _env):
            return "".join(
                getattr(t, "content", "") for t in tokens if t.type == "inline"
            )

    class FakeMarkdown:
        renderer = FakeRenderer()
        options = {}

        def parse(self, _text):
            return tokens

    html, headings = mdp.render_markdown_with_headings(FakeMarkdown(), "", {})
    assert "First" in html and len(headings) == 4
    assert isinstance(tokens[0].attrs, list) and tokens[0].attrs[0] == [
        "id",
        "first",
    ]
    assert tokens[2].attrs["id"] == "second"
    assert tokens[4].attrs[0][1] == "third"
    assert tokens[6].attrs[-1] == ["id", "fourth"]


def test_build_toc_html_filters_levels() -> None:
    headings = [
        mdp.Heading(level=1, text="A", anchor="a"),
        mdp.Heading(level=1, text="B", anchor="b"),
        mdp.Heading(level=3, text="Too deep", anchor="deep"),
    ]
    assert mdp.build_toc_html(headings, max_depth=0) == ""
    toc = mdp.build_toc_html(headings, max_depth=2)
    assert "A" in toc and "Too deep" not in toc and "B" in toc


def test_build_toc_html_closes_nested_lists() -> None:
    headings = [
        mdp.Heading(level=1, text="Intro", anchor="intro"),
        mdp.Heading(level=2, text="Part", anchor="part"),
        mdp.Heading(level=3, text="Deep", anchor="deep"),
    ]
    toc = mdp.build_toc_html(headings, max_depth=3)
    assert "</ul></li>" in toc
    assert toc.endswith("</ul>")


def test_generate_ai_title_fields_regex_fallback(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    module = ModuleType("study_utils.transcribe_video")

    def fake_load_client():
        stub = openai_factory()
        stub.queue_response('title: "Loose Title"')
        return stub

    module.load_client = fake_load_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "study_utils.transcribe_video", module)
    fields = mdp.generate_ai_title_fields(sample_text="Sample")
    assert fields.title == "Loose Title"


def test_load_default_css_path_handles_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "print.css":
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert mdp.load_default_css_path() is None


def test_resolve_base_url_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path.cwd())
    args = Namespace(resources=None)
    url = mdp._resolve_base_url(args)
    assert url.startswith("file:")


def test_build_stylesheets_without_custom_css() -> None:
    class DummyCSS:
        def __init__(self, *, string=None, filename=None):
            self.string = string
            self.filename = filename

    page_css = "@page {}"
    args = Namespace(css=None)
    sheets = mdp._build_stylesheets(args, DummyCSS, page_css)
    assert len(sheets) == 1 and sheets[0].string == page_css


def test_iter_markdown_files_skips_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = Path("/path/does/not/exist")
    assert list(iter_markdown_files([missing], level_limit=0)) == []


def test_default_highlight_css() -> None:
    css = mdp.default_highlight_css()
    assert ".highlight" in css


def test_generate_ai_title_fields_success(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    module = ModuleType("study_utils.transcribe_video")

    def fake_load_client():
        stub = openai_factory()
        stub.queue_response(
            '{"title": "AI Title", "subtitle": "AI Sub", "author": "AI"}'
        )
        return stub

    module.load_client = fake_load_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "study_utils.transcribe_video", module)

    fields = mdp.generate_ai_title_fields(sample_text="Sample")
    assert fields.title == "AI Title"
    assert fields.subtitle == "AI Sub"
    assert fields.author == "AI"
    assert fields.date_str


def test_generate_ai_title_fields_handles_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("study_utils.transcribe_video")

    def bad_client():
        raise RuntimeError("no client")

    module.load_client = bad_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "study_utils.transcribe_video", module)
    fields = mdp.generate_ai_title_fields(sample_text="Sample")
    assert fields == TitleFields()


def test_generate_ai_title_fields_chat_failure(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    module = ModuleType("study_utils.transcribe_video")

    def fake_load_client():
        class Client:
            class Chat:
                class Completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("nope")

                completions = Completions()

            chat = Chat()

        return Client()

    module.load_client = fake_load_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "study_utils.transcribe_video", module)
    fields = mdp.generate_ai_title_fields(sample_text="Sample")
    assert fields == TitleFields()


def test_generate_ai_title_fields_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("transcribe_video") or (
            name == "study_utils" and "transcribe_video" in fromlist
        ):
            raise ImportError("boom")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    fields = mdp.generate_ai_title_fields(sample_text="Sample")
    assert fields == TitleFields()

    module = ModuleType("study_utils.transcribe_video")

    def bad_client():
        raise RuntimeError("no client")

    module.load_client = bad_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "study_utils.transcribe_video", module)
    fields = mdp.generate_ai_title_fields(sample_text="Sample")
    assert fields == TitleFields()


def test_load_default_css_path_exists() -> None:
    path = mdp.load_default_css_path()
    assert path is None or path.exists()


def test_assemble_html_includes_parts(tmp_path: Path) -> None:
    parts = [("Intro", "<p>Hello</p>"), ("Details", "<p>World</p>")]
    html = mdp.assemble_html(
        parts,
        include_toc=True,
        toc_depth=2,
        highlight_css=".highlight { }",
        custom_css_href="custom.css",
        title_html="<header>Title</header>",
    )
    assert "Title" in html and "custom.css" in html
    assert "Table of Contents" in html


def test_build_markdown_it_enables_unknown_extension_gracefully() -> None:
    md = build_markdown_it(["unknown-plugin"])
    assert md.parse("# Title")


def test_parse_markdown_args_parses_defaults() -> None:
    args = mdp._parse_markdown_args(["out.pdf", "doc.md"])
    assert args.OUTPUT == "out.pdf"
    assert args.INPUTS == ["doc.md"]
    assert args.paper_size == "letter"


def test_collect_markdown_inputs_and_empty(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.md"
    file_path.write_text("# Doc", encoding="utf-8")
    args = Namespace(
        INPUTS=[str(tmp_path)], extensions=[], level_limit=0, sort="name"
    )
    files = mdp._collect_markdown_inputs(args)
    assert files == [file_path]

    args_empty = Namespace(
        INPUTS=[str(tmp_path / "missing")],
        extensions=[],
        level_limit=0,
        sort="name",
    )
    with pytest.raises(SystemExit):
        mdp._collect_markdown_inputs(args_empty)


def test_build_page_css_from_args_handles_errors() -> None:
    args = Namespace(
        paper_size="letter",
        orientation="portrait",
        margin=None,
        margin_top=None,
        margin_right=None,
        margin_bottom=None,
        margin_left=None,
    )
    css = mdp._build_page_css_from_args(args)
    assert "Letter" in css

    args_bad = Namespace(
        paper_size="letter",
        orientation="upside-down",
        margin=None,
        margin_top=None,
        margin_right=None,
        margin_bottom=None,
        margin_left=None,
    )
    with pytest.raises(SystemExit):
        mdp._build_page_css_from_args(args_bad)


def test_render_markdown_parts_collects_samples(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.md"
    file_path.write_text("# Doc\n\nContent", encoding="utf-8")
    md = build_markdown_it([])
    parts, sample = mdp._render_markdown_parts([file_path], md)
    assert len(parts) == 1
    assert "Content" in sample


def test_build_title_page_html_merges_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(
        title_page=True,
        title="",
        subtitle="",
        author="",
        date="",
        ai_title=True,
        ai_model="model",
        ai_max_tokens=100,
        ai_temperature=0.1,
        title_template=None,
    )
    ai_fields = TitleFields(
        title="AI Title", subtitle="AI Sub", author="AI", date_str="2024-01-01"
    )
    monkeypatch.setattr(mdp, "generate_ai_title_fields", lambda **_: ai_fields)
    html = mdp._build_title_page_html(args, sample_text="Hello")
    assert "AI Title" in html and "AI Sub" in html


def test_print_dry_run_outputs_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = Namespace(
        paper_size="letter",
        orientation="portrait",
        title_page=True,
        ai_title=True,
        toc=True,
        toc_depth=2,
        name="quiz",
    )
    files = [tmp_path / "a.md", tmp_path / "b.md"]
    out = tmp_path / "out.pdf"
    mdp._print_dry_run(args, files, out)
    captured = capsys.readouterr()
    assert "Planned output" in captured.out and "a.md" in captured.out


def test_load_weasyprint_uses_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    html_cls, css_cls = mdp._load_weasyprint()
    assert html_cls.__name__ == "HTMLStub"  # from fixtures stub
    assert css_cls.__name__ == "CSSStub"

    monkeypatch.setitem(sys.modules, "weasyprint", None)
    with pytest.raises(SystemExit):
        mdp._load_weasyprint()


def test_resolve_base_url(tmp_path: Path) -> None:
    args = Namespace(resources=str(tmp_path))
    url = mdp._resolve_base_url(args)
    assert url.startswith("file:")


def test_build_stylesheets(tmp_path: Path) -> None:
    class DummyCSS:
        def __init__(self, *, string=None, filename=None):
            self.string = string
            self.filename = filename

    page_css = "@page {}"
    args = Namespace(css=str(tmp_path / "styles.css"))
    (tmp_path / "styles.css").write_text("", encoding="utf-8")
    sheets = mdp._build_stylesheets(args, DummyCSS, page_css)
    assert len(sheets) == 2
    assert sheets[0].string == page_css
    assert sheets[1].filename == str((tmp_path / "styles.css").resolve())


def test_main_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("# Doc", encoding="utf-8")

    monkeypatch.setattr(
        mdp, "generate_ai_title_fields", lambda **_: TitleFields()
    )

    argv = [
        "--dry-run",
        "--title-page",
        "--toc",
        str(tmp_path / "out.pdf"),
        str(doc),
    ]
    mdp.main(argv)
    captured = capsys.readouterr()
    assert "Planned output" in captured.out


def test_main_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("# Doc", encoding="utf-8")

    monkeypatch.setattr(
        mdp, "generate_ai_title_fields", lambda **_: TitleFields()
    )

    argv = [
        "--verbose",
        str(tmp_path / "out.pdf"),
        str(doc),
    ]
    mdp.main(argv)

    # ensure stub wrote PDF using the installed weasyprint stub
    html_cls = sys.modules["weasyprint"].HTML
    calls = html_cls.pop_calls()
    assert calls and calls[0].target == tmp_path / "out.pdf"


def test_main_handles_generation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("# Doc", encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mdp, "assemble_html", boom)
    argv = [str(tmp_path / "out.pdf"), str(doc)]
    with pytest.raises(RuntimeError):
        mdp.main(argv)
