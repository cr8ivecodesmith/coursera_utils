from __future__ import annotations

import os
from pathlib import Path

import study_utils.text_combiner as tc


def test_parse_extensions_defaults_and_normalization():
    assert tc.parse_extensions(None) == {"txt"}
    assert tc.parse_extensions([".txt", "MD"]) == {"txt", "md"}


def test_iter_text_files_with_level_limit_and_extensions(tmp_path: Path):
    # layout:
    # root/
    #   a.txt, b.md, c.bin
    #   sub/
    #     d.txt
    #     deeper/
    #       e.txt
    a = tmp_path / "a.txt"
    a.write_text("A")
    (tmp_path / "b.md").write_text("B")
    (tmp_path / "c.bin").write_text("X")
    sub = tmp_path / "sub"
    sub.mkdir()
    d = sub / "d.txt"
    d.write_text("D")
    deeper = sub / "deeper"
    deeper.mkdir()
    e = deeper / "e.txt"
    e.write_text("E")

    # Default extensions -> only txt
    files = list(tc.iter_text_files([tmp_path], {"txt"}, level_limit=0))
    assert set(files) == {a, d, e}

    # Limit to 2 parts (dir+filename) -> include a and d but not e
    files_lvl2 = list(tc.iter_text_files([tmp_path], {"txt"}, level_limit=2))
    assert set(files_lvl2) == {a, d}

    files_lvl1 = list(tc.iter_text_files([tmp_path], {"txt"}, level_limit=1))
    assert set(files_lvl1) == {a}

    # Multiple extensions
    files_md = list(tc.iter_text_files([tmp_path], {"txt", "md"}, level_limit=0))
    assert set(files_md) == {a, d, e, tmp_path / "b.md"}


def test_order_files_by_name_and_modified(tmp_path: Path):
    f1 = tmp_path / "b.txt"
    f2 = tmp_path / "a.txt"
    f1.write_text("B")
    f2.write_text("A")

    # name ascending
    ordered = tc.order_files([f1, f2], "name")
    assert ordered == [f2, f1]

    # modified ascending: set mtimes explicit
    os.utime(f1, (111111111, 111111111))
    os.utime(f2, (222222222, 222222222))
    ordered_mod = tc.order_files([f1, f2], "modified")
    assert ordered_mod == [f1, f2]

    # modified descending
    ordered_mod_desc = tc.order_files([f1, f2], "-modified")
    assert ordered_mod_desc == [f2, f1]


def test_combine_by_new_and_eof(tmp_path: Path):
    p1 = tmp_path / "x.txt"
    p2 = tmp_path / "y.txt"
    p1.write_text("one")
    p2.write_text("two")
    out_new = tmp_path / "out_new.txt"
    out_eof = tmp_path / "out_eof.txt"

    opts_new = tc.CombineOptions(
        extensions={"txt"},
        level_limit=0,
        combine_by="NEW",
        order_by=None,
        section_title=None,
        section_title_format=None,
        section_title_heading=None,
    )
    opts_eof = tc.CombineOptions(
        extensions={"txt"},
        level_limit=0,
        combine_by="EOF",
        order_by=None,
        section_title=None,
        section_title_format=None,
        section_title_heading=None,
    )

    tc.combine_files([p1, p2], out_new, opts_new)
    tc.combine_files([p1, p2], out_eof, opts_eof)

    assert out_new.read_text() == "one\n" + "two"
    assert out_eof.read_text() == "one" + "two"


def test_section_title_filename_with_format_and_heading(tmp_path: Path):
    p1 = tmp_path / "alpha beta.txt"
    p2 = tmp_path / "gamma.txt"
    p1.write_text("AAA")
    p2.write_text("BBB")
    outp = tmp_path / "out.md"

    opts = tc.CombineOptions(
        extensions={"txt"},
        level_limit=0,
        combine_by="NEW",
        order_by=None,
        section_title="filename",
        section_title_format="title",
        section_title_heading="##",
    )

    tc.combine_files([p1, p2], outp, opts)
    expected = (
        "## Alpha Beta\n\n"  # first section title (no leading blank line)
        "AAA"
        "\n"
        "\n## Gamma\n\n"  # second section title preceded by one blank line
        "BBB"
    )
    assert outp.read_text() == expected


def test_cli_basic_combine(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A")
    b.write_text("B")
    outp = tmp_path / "combined.txt"

    # Order by name so result is deterministic regardless of discovery order
    tc.main([str(outp), str(tmp_path), "--extensions", "txt", "--order-by", "name", "--combine-by", "NEW", "--level-limit", "1"])

    assert outp.exists()
    # a then b due to name ordering and newline separator
    assert outp.read_text() == "A\nB"
