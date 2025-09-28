from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast

from study_utils.convert_markdown import config as cfg
from study_utils.convert_markdown import converter


def _deps(
    markdown: Callable[[Path], str],
    epub: Callable[[Path], str],
) -> converter.ConverterDependencies:
    return converter.ConverterDependencies(
        markitdown=markdown,
        epub=epub,
    )


def _fixed_now() -> datetime:
    return datetime(2025, 3, 18, 14, 0, tzinfo=timezone.utc)


def _touch(path: Path, when: datetime) -> None:
    timestamp = when.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_convert_file_markitdown_success(tmp_path):
    source = tmp_path / "input.pdf"
    source.write_text("dummy", encoding="utf-8")
    _touch(source, datetime(2025, 3, 17, 10, tzinfo=timezone.utc))

    output_dir = tmp_path / "out"
    markdown_body = "# Title\nConverted content\n"

    deps = _deps(lambda _: markdown_body, lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=output_dir,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.SUCCESS
    assert result.output_path == output_dir / "input.md"
    text = result.output_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert 'source_path: "' in text
    assert 'converted_at: "2025-03-18T14:00:00Z"' in text
    assert "# Title" in text


def test_convert_file_epub_uses_fallback(tmp_path):
    source = tmp_path / "chapter.epub"
    source.write_text("epub", encoding="utf-8")
    output_dir = tmp_path / "out"

    def fail_markdown(_: Path) -> str:  # pragma: no cover - guard
        raise AssertionError("markitdown should not run")

    deps = _deps(fail_markdown, lambda _: "EPUB body")

    result = converter.convert_file(
        source,
        output_dir=output_dir,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.SUCCESS
    assert result.output_path == output_dir / "chapter.md"
    assert "EPUB body" in result.output_path.read_text(encoding="utf-8")


def test_convert_file_skip_on_collision(tmp_path):
    source = tmp_path / "report.pdf"
    source.write_text("content", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    target = output_dir / "report.md"
    target.write_text("existing", encoding="utf-8")

    deps = _deps(lambda _: "ignored", lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=output_dir,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.SKIPPED
    assert result.output_path == target
    assert target.read_text(encoding="utf-8") == "existing"


def test_convert_file_overwrite(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("notes", encoding="utf-8")
    output_dir = tmp_path / "out"
    target = output_dir / "notes.md"
    target.parent.mkdir()
    target.write_text("old", encoding="utf-8")

    deps = _deps(lambda _: "fresh", lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=output_dir,
        collision=cfg.CollisionPolicy.OVERWRITE,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.SUCCESS
    assert target.read_text(encoding="utf-8").endswith("fresh\n")


def test_convert_file_versions_existing_outputs(tmp_path):
    source = tmp_path / "summary.html"
    source.write_text("summary", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "summary.md").write_text("v0", encoding="utf-8")
    (output_dir / "summary-01.md").write_text("v1", encoding="utf-8")

    deps = _deps(lambda _: "new", lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=output_dir,
        collision=cfg.CollisionPolicy.VERSION,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.SUCCESS
    assert result.output_path == output_dir / "summary-02.md"
    assert result.output_path.read_text(encoding="utf-8").endswith("new\n")


def test_convert_file_missing_source_reports_failure(tmp_path):
    source = tmp_path / "missing.pdf"
    deps = _deps(lambda _: "", lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=tmp_path,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.FAILED
    assert "not found" in (result.reason or "")
    assert result.output_path is None


def test_convert_file_unsupported_extension(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("data", encoding="utf-8")
    deps = _deps(lambda _: "", lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=tmp_path,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.FAILED
    assert "Unsupported file extension" in (result.reason or "")


def test_convert_file_backend_failure(tmp_path):
    source = tmp_path / "failure.docx"
    source.write_text("doc", encoding="utf-8")

    def failing_markdown(_: Path) -> str:
        raise ValueError("boom")

    deps = _deps(failing_markdown, lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=tmp_path,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.FAILED
    assert result.reason == "boom"


def test_convert_file_missing_extension(tmp_path):
    source = tmp_path / "noext"
    source.write_text("text", encoding="utf-8")
    deps = _deps(lambda _: "", lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=tmp_path,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.FAILED
    assert "not supported" in (result.reason or "")


def test_convert_file_unknown_collision_policy(tmp_path):
    source = tmp_path / "file.pdf"
    source.write_text("body", encoding="utf-8")
    target = tmp_path / "file.md"
    target.write_text("existing", encoding="utf-8")
    deps = _deps(lambda _: "ok", lambda _: "")

    bogus_policy = cast(cfg.CollisionPolicy, object())

    result = converter.convert_file(
        source,
        output_dir=tmp_path,
        collision=bogus_policy,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.FAILED
    assert "Unsupported collision policy" in (result.reason or "")


def test_convert_file_rejects_directory_source(tmp_path):
    source_dir = tmp_path / "dir"
    source_dir.mkdir()
    deps = _deps(lambda _: "", lambda _: "")

    result = converter.convert_file(
        source_dir,
        output_dir=tmp_path,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=_fixed_now,
    )

    assert result.status is converter.ConversionStatus.FAILED
    assert "not a file" in (result.reason or "")


def test_convert_file_uses_default_now(tmp_path):
    source = tmp_path / "auto.pdf"
    source.write_text("data", encoding="utf-8")
    deps = _deps(lambda _: "body", lambda _: "")

    result = converter.convert_file(
        source,
        output_dir=tmp_path,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
    )

    assert result.status is converter.ConversionStatus.SUCCESS
    text = result.output_path.read_text(encoding="utf-8")
    assert "converted_at: " in text
    assert text.endswith("body\n")


def test_convert_file_normalizes_naive_timestamp(tmp_path):
    source = tmp_path / "naive.pdf"
    source.write_text("data", encoding="utf-8")
    deps = _deps(lambda _: "body", lambda _: "")

    naive_now = datetime(2025, 3, 19, 8, 45)

    result = converter.convert_file(
        source,
        output_dir=tmp_path,
        collision=cfg.CollisionPolicy.SKIP,
        dependencies=deps,
        now=lambda: naive_now,
    )

    text = result.output_path.read_text(encoding="utf-8")
    assert 'converted_at: "2025-03-19T08:45:00Z"' in text
