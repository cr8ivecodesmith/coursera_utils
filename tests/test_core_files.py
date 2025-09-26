from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from study_utils.core import (
    iter_text_files,
    order_files,
    parse_extensions,
    read_text_file,
)
from study_utils.core.files import _within_level_limit


def test_parse_extensions_default_uses_custom_fallback() -> None:
    assert parse_extensions(None, default={"md"}) == {"md"}


def test_parse_extensions_strips_dots_and_lowercases() -> None:
    result = parse_extensions([".TXT", "Md"], default={"txt"})
    assert result == {"txt", "md"}


def test_parse_extensions_ignores_non_strings_and_returns_fallback() -> None:
    result = parse_extensions([" ", 123, None], default={"rst"})
    assert result == {"rst"}


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


def test_iter_text_files_requires_nonnegative_level_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        list(iter_text_files([tmp_path], {"txt"}, level_limit=-1))


def test_iter_text_files_errors_on_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        list(iter_text_files([missing], {"txt"}, level_limit=0))



def test_iter_text_files_yields_matching_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hi", encoding="utf-8")
    result = list(iter_text_files([target], {"txt"}, level_limit=0))
    assert result == [target]

def test_iter_text_files_skips_non_matching_extensions(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("hi", encoding="utf-8")
    result = list(iter_text_files([target], {"txt"}, level_limit=0))
    assert result == []


def test_iter_text_files_skips_entries_not_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    special = tmp_path / "special"
    special.mkdir()

    original_is_dir = Path.is_dir
    original_is_file = Path.is_file

    def fake_is_dir(self: Path) -> bool:
        if self == special:
            return False
        return original_is_dir(self)

    def fake_is_file(self: Path) -> bool:
        if self == special:
            return False
        return original_is_file(self)

    monkeypatch.setattr(Path, "is_dir", fake_is_dir)
    monkeypatch.setattr(Path, "is_file", fake_is_file)

    result = list(iter_text_files([special], {"txt"}, level_limit=0))
    assert result == []


def test_within_level_limit_handles_unrelated_paths(tmp_path: Path) -> None:
    outside = tmp_path.parent
    assert _within_level_limit(outside, tmp_path, level_limit=1) is False


def test_order_files_by_modified(tmp_path: Path) -> None:
    newer = tmp_path / "newer.txt"
    older = tmp_path / "older.txt"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")

    os.utime(older, (0, 1))
    os.utime(newer, (0, 2))

    ordered = order_files([older, newer], "modified")
    assert ordered == [older, newer]

    ordered_desc = order_files([older, newer], "-modified")
    assert ordered_desc == [newer, older]


def test_order_files_by_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    stats = {
        first: SimpleNamespace(st_birthtime=1, st_ctime=1),
        second: SimpleNamespace(st_birthtime=2, st_ctime=2),
    }

    def fake_stat(self: Path):
        return stats[self]

    monkeypatch.setattr(Path, "stat", fake_stat)

    ordered = order_files([second, first], "created")
    assert ordered == [first, second]


def test_order_files_handles_stat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.txt"
    target.write_text("a", encoding="utf-8")

    original_stat = Path.stat

    def fake_stat(self: Path):
        if self == target:
            raise OSError("boom")
        return original_stat(self)

    monkeypatch.setattr(Path, "stat", fake_stat)

    ordered = order_files([target], "modified")
    assert ordered == [target]


def test_order_files_returns_copy_when_unsorted(tmp_path: Path) -> None:
    paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for p in paths:
        p.write_text(p.name, encoding="utf-8")
    result = order_files(paths, None)
    assert result == paths
    assert result is not paths


def test_order_files_unknown_key_falls_back(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("1", encoding="utf-8")
    second.write_text("2", encoding="utf-8")
    ordered = order_files([first, second], "size")
    assert ordered == [first, second]


def test_read_text_file_round_trips_content(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("hello", encoding="utf-8")
    assert read_text_file(target) == "hello"
