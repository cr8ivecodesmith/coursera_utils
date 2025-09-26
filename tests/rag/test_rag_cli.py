from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from study_utils import cli as root_cli
from study_utils.rag import cli as rag_cli
from study_utils.rag import config as config_mod


def test_rag_command_registered():
    assert "rag" in root_cli.COMMANDS


def test_config_init_creates_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    target = tmp_path / "custom" / "rag.toml"
    exit_code = rag_cli.main(["config", "init", "--path", str(target)])

    assert exit_code == 0
    assert target.exists()
    out = capsys.readouterr().out
    assert "Wrote config template" in out


def test_config_init_requires_force(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    target = tmp_path / "rag.toml"
    config_mod.write_template(target)

    exit_code = rag_cli.main(["config", "init", "--path", str(target)])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Config already exists" in err


def test_config_validate_reports_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "dh"))
    target = tmp_path / "rag.toml"
    config_mod.write_template(target)

    exit_code = rag_cli.main(["config", "validate", "--path", str(target)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Configuration OK" in output
    assert "chunk_tokens" in output


def test_config_validate_propagates_errors(tmp_path, capsys):
    target = tmp_path / "bad.toml"
    target.write_text("not valid", encoding="utf-8")

    exit_code = rag_cli.main(["config", "validate", "--path", str(target)])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Failed to parse" in err


def test_config_path_prints_resolved_value(tmp_path, capsys):
    target = tmp_path / "custom" / "rag.toml"
    exit_code = rag_cli.main(["config", "path", "--path", str(target)])

    assert exit_code == 0
    output = capsys.readouterr().out.strip()
    assert Path(output) == target.resolve()


def test_config_init_uses_default_path(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    exit_code = rag_cli.main(["config", "init", "--force"])
    assert exit_code == 0


def test_handle_config_unknown_command():
    with pytest.raises(RuntimeError):
        rag_cli._handle_config(argparse.Namespace(config_command="oops"))


def test_config_path_handles_resolve_error(monkeypatch, capsys):
    def raise_error(**kwargs):  # noqa: D401, ARG001
        raise config_mod.ConfigError("boom")

    monkeypatch.setattr(config_mod, "resolve_config_path", raise_error)
    exit_code = rag_cli.main(["config", "path"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "boom" in err


def test_main_reports_unhandled_command(monkeypatch):
    class DummyParser:
        def __init__(self):
            self.error_message = None

        def parse_args(self, argv=None):  # noqa: ARG002
            return argparse.Namespace(command="todo")

        def error(self, message):
            self.error_message = message

    dummy = DummyParser()
    monkeypatch.setattr(rag_cli, "_build_parser", lambda: dummy)

    exit_code = rag_cli.main(["todo"])
    assert exit_code == 2
    assert dummy.error_message == "Command not implemented yet."


def test_to_path_none_returns_none():
    assert rag_cli._to_path(None) is None
