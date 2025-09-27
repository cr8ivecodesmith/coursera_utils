from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import study_utils.generate_document as gd


def test_find_config_path_with_custom_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        gd.find_config_path(str(missing))


def test_find_config_path_with_explicit_existing(tmp_path: Path) -> None:
    cfg = tmp_path / "custom.toml"
    cfg.write_text("[doc]\nprompt='Use'\n", encoding="utf-8")
    result = gd.find_config_path(str(cfg))
    assert result == cfg.resolve()


def test_load_documents_config_empty_file(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.toml"
    cfg.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        gd.load_documents_config(cfg)


def test_load_documents_config_mixed_entries(tmp_path: Path) -> None:
    cfg = tmp_path / "mixed.toml"
    cfg.write_text(
        "invalid = 1\n[skip]\nprompt = ''\n[ok]\nprompt = 'Do this'\n",
        encoding="utf-8",
    )
    data = gd.load_documents_config(cfg)
    assert "ok" in data and data["ok"]["prompt"] == "Do this"


def test_load_documents_config_import_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "doc.toml"
    cfg.write_text("[doc]\nprompt='Use me'\n", encoding="utf-8")
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"tomllib", "tomli"}:
            raise ImportError("boom")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError):
        gd.load_documents_config(cfg)


def test_find_config_path_prefers_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "documents.toml"
    cfg.write_text("[doc]\nprompt='Use me'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert gd.find_config_path(None) == cfg.resolve()


def test_find_config_path_raises_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_exists(self: Path) -> bool:
        calls.append(self)
        return False

    monkeypatch.setattr(Path, "exists", fake_exists)
    with pytest.raises(FileNotFoundError):
        gd.find_config_path(None)
    assert calls  # ensure our stub ran


def test_load_documents_config_filters_invalid_entries(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text("[bad]\ndescription='Only description'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        gd.load_documents_config(cfg)


def test_build_reference_block_and_messages(tmp_path: Path) -> None:
    files = [
        (tmp_path / "a.txt", "Alpha"),
        (tmp_path / "b.md", "Beta"),
    ]
    block = gd.build_reference_block(files)
    assert "File: a.txt" in block and "Beta" in block

    cfg = {"prompt": "Write summary", "model": "gpt-4o-mini", "description": ""}
    messages = gd.build_messages(cfg, files)
    assert messages[0]["role"] == "system"
    assert "Write summary" in messages[1]["content"]
    assert "Alpha" in messages[1]["content"]


def test_generate_document_writes_output_with_stubbed_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.md"
    a.write_text("Alpha content", encoding="utf-8")
    b.write_text("Beta content", encoding="utf-8")

    # Prepare stub response
    openai_factory.reset()
    stub = openai_factory()
    stub.queue_response("# Result\n\nGenerated.")
    monkeypatch.setattr(gd, "load_client", lambda: stub)

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
    assert out.read_text(encoding="utf-8").startswith("# Result")
    assert stub.calls  # ensure client was exercised


def test_generate_document_unknown_type_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "x.txt"
    p.write_text("X", encoding="utf-8")
    monkeypatch.setattr(gd, "load_client", lambda: object())

    with pytest.raises(ValueError):
        gd.generate_document(
            doc_type="does_not_exist",
            output_path=tmp_path / "out.md",
            inputs=[tmp_path],
            extensions={"txt"},
            level_limit=0,
            config_path=gd.find_config_path(None),
        )


def test_generate_document_no_matching_files_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.bin").write_text("X", encoding="utf-8")
    monkeypatch.setattr(gd, "load_client", lambda: object())
    with pytest.raises(FileNotFoundError):
        gd.generate_document(
            doc_type="keywords",
            output_path=tmp_path / "out.md",
            inputs=[tmp_path],
            extensions={"txt"},
            level_limit=0,
            config_path=gd.find_config_path(None),
        )


def test_generate_document_raises_when_ai_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref_file = tmp_path / "a.txt"
    ref_file.write_text("data", encoding="utf-8")

    class EmptyClient:
        def __init__(self) -> None:
            message = SimpleNamespace(content=" ")
            choice = SimpleNamespace(message=message)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(choices=[choice])
                )
            )

    monkeypatch.setattr(gd, "load_client", EmptyClient)
    with pytest.raises(RuntimeError):
        gd.generate_document(
            doc_type="keywords",
            output_path=tmp_path / "out.md",
            inputs=[tmp_path],
            extensions={"txt"},
            level_limit=0,
            config_path=gd.find_config_path(None),
        )


def test_main_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    openai_factory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    ref = src_dir / "ref.txt"
    ref.write_text("Reference", encoding="utf-8")

    stub = openai_factory()
    stub.queue_response("# Title\n\nBody")
    monkeypatch.setattr(gd, "load_client", lambda: stub)
    out_path = tmp_path / "out.md"
    argv = [
        "keywords",
        str(out_path),
        str(src_dir),
    ]
    gd.main(argv)
    captured = capsys.readouterr()
    assert "Generated document" in captured.out
    assert out_path.exists()


def test_main_handles_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "keywords",
        str(tmp_path / "out.md"),
        str(tmp_path),
        "--level-limit",
        "-1",
    ]
    with pytest.raises(SystemExit) as exc:
        gd.main(argv)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.out


def test_main_handles_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ref = tmp_path / "ref.txt"
    ref.write_text("ref", encoding="utf-8")

    def fake_generate(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gd, "generate_document", fake_generate)
    argv = ["keywords", str(tmp_path / "out.md"), str(tmp_path)]
    with pytest.raises(SystemExit) as exc:
        gd.main(argv)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Failed to generate document" in captured.out
