from __future__ import annotations

from study_utils.workspace import cli


def test_study_init_creates_workspace(tmp_path, capsys, monkeypatch):
    target = tmp_path / "workspace"
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(target))

    code = cli.main([])

    captured = capsys.readouterr()
    assert code == 0
    assert "Workspace ready" in captured.out
    assert target.is_dir()


def test_study_init_supports_custom_path(tmp_path, capsys):
    target = tmp_path / "custom"

    code = cli.main(["--path", str(target)])

    captured = capsys.readouterr()
    assert code == 0
    assert target.is_dir()
    assert str(target) in captured.out


def test_study_init_quiet_mode(tmp_path, capsys, monkeypatch):
    target = tmp_path / "quiet"
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(target))

    code = cli.main(["--quiet"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
