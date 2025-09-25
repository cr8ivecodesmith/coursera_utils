from __future__ import annotations

from pathlib import Path

import study_utils.generate_document as gd


def test_parse_extensions_defaults_and_normalization():
    assert gd.parse_extensions(None) == {"txt", "md", "markdown"}
    assert gd.parse_extensions([".TXT", "Md"]) == {"txt", "md"}


def test_load_documents_config_bundled_and_find_path():
    # Should find bundled study_utils/documents.toml when none is provided
    cfg_path = gd.find_config_path(None)
    cfg = gd.load_documents_config(cfg_path)
    # Ensure expected defaults exist
    assert {"keywords", "reading_assignment", "book"}.issubset(set(cfg.keys()))
    for k in ["keywords", "reading_assignment", "book"]:
        assert isinstance(cfg[k]["prompt"], str) and cfg[k]["prompt"].strip()


def test_generate_document_writes_output_with_stubbed_client(tmp_path: Path, monkeypatch):
    # Create reference files
    a = tmp_path / "a.txt"
    b = tmp_path / "b.md"
    a.write_text("Alpha content")
    b.write_text("Beta content")

    # Stub load_client to avoid env and network
    class _Msg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    # Return fixed markdown so we can assert output
                    return _Resp("# Result\n\nGenerated.")

    monkeypatch.setattr(gd, "load_client", lambda: FakeClient())

    out = tmp_path / "out.md"
    used = gd.generate_document(
        doc_type="keywords",
        output_path=out,
        inputs=[tmp_path],
        extensions={"txt", "md"},
        level_limit=0,
        config_path=gd.find_config_path(None),
    )
    assert used == 2
    assert out.read_text().startswith("# Result")


def test_generate_document_unknown_type_raises(tmp_path: Path, monkeypatch):
    # Provide at least one file
    p = tmp_path / "x.txt"
    p.write_text("X")

    # Stub client even though we expect early error
    monkeypatch.setattr(gd, "load_client", lambda: object())

    out = tmp_path / "out.md"
    try:
        gd.generate_document(
            doc_type="does_not_exist",
            output_path=out,
            inputs=[tmp_path],
            extensions={"txt"},
            level_limit=0,
            config_path=gd.find_config_path(None),
        )
    except ValueError as e:
        assert "Unknown document type" in str(e)
    else:
        raise AssertionError("Expected ValueError for unknown doc type")


def test_generate_document_no_matching_files_raises(tmp_path: Path, monkeypatch):
    # Directory with only .bin file
    (tmp_path / "a.bin").write_text("X")
    monkeypatch.setattr(gd, "load_client", lambda: object())
    out = tmp_path / "out.md"
    try:
        gd.generate_document(
            doc_type="keywords",
            output_path=out,
            inputs=[tmp_path],
            extensions={"txt"},
            level_limit=0,
            config_path=gd.find_config_path(None),
        )
    except FileNotFoundError as e:
        assert "No matching reference files" in str(e)
    else:
        raise AssertionError("Expected FileNotFoundError when no files matched")
