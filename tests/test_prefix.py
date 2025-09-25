from pathlib import Path

import study_utils.transcribe_video as tv


def test_parse_prefix_and_build_counter_widths():
    parts = tv.parse_prefix_parts(
        [
            "text:PRE-",
            "counter:N",
            "text:-",
            "counter:NNNN",
        ]
    )
    # Index 3 should render as: PRE-3-0003
    s = tv.build_prefix_string(parts, 3)
    assert s == "PRE-3-0003"


def test_legacy_sep_treated_as_text():
    parts = tv.parse_prefix_parts(["text:A", "sep:-", "counter:NN"])
    s = tv.build_prefix_string(parts, 7)
    assert s == "A-07"


def test_make_output_filename_with_smart_base_and_prefix():
    parts = tv.parse_prefix_parts(["text:This-", "counter:NN", "text:-"])
    out = tv.make_output_filename(
        Path("x.mp4"), 12, parts, smart_base="Intro to ML"
    )
    # zero-padded 2 digits for NN => 12 -> 12
    assert out == "This-12-Intro to ML.txt"
