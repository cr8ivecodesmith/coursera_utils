from __future__ import annotations

from pathlib import Path

import os

from study_utils.core import (
    iter_text_files,
    order_files,
    parse_extensions,
    read_text_file,
)


def test_parse_extensions_default_uses_custom_fallback() -> None:
    assert parse_extensions(None, default={"md"}) == {"md"}


def test_parse_extensions_strips_dots_and_lowercases() -> None:
    result = parse_extensions([".TXT", "Md"], default={"txt"})
    assert result == {"txt", "md"}


def test_iter_text_files_respects_level_limit(tmp_path: Path) -> None:
    top = tmp_path / "top.txt"
    top.write_text("top", encoding="utf-8")

    nested_dir = tmp_path / "dir"
    nested_dir.mkdir()
    nested = nested_dir / "nested.txt"
    nested.write_text("nested", encoding="utf-8")

    all_files = [
        f.name for f in iter_text_files([tmp_path], {"txt"}, level_limit=0)
    ]
    assert all_files == ["nested.txt", "top.txt"]

    shallow_files = [
        f.name for f in iter_text_files([tmp_path], {"txt"}, level_limit=1)
    ]
    assert shallow_files == ["top.txt"]


def test_order_files_by_modified(tmp_path: Path) -> None:
    newer = tmp_path / "newer.txt"
    older = tmp_path / "older.txt"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")

    # Ensure modified times differ
    os.utime(older, (0, 1))
    os.utime(newer, (0, 2))

    ordered = order_files([older, newer], "modified")
    assert ordered == [older, newer]

    ordered_desc = order_files([older, newer], "-modified")
    assert ordered_desc == [newer, older]


def test_read_text_file_round_trips_content(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("hello", encoding="utf-8")
    assert read_text_file(target) == "hello"
