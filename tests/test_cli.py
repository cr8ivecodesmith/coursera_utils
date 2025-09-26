import pytest

from study_utils import cli


@pytest.fixture(autouse=True)
def reset_metadata(monkeypatch):
    """Ensure metadata.version is controllable during tests."""

    def fake_version(name: str) -> str:
        assert name == "study-utils"
        return "0.0-test"

    monkeypatch.setattr(cli.metadata, "version", fake_version)
    yield


def test_version_command_handles_missing_package(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(cli.metadata.PackageNotFoundError()),
    )
    code = cli.main(["version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "unknown"


def test_no_args_prints_usage_and_returns_error(capsys):
    code = cli.main([])
    captured = capsys.readouterr()
    assert code == 2
    assert "Usage: study" in captured.out
    assert "Available commands:" in captured.out


def test_help_flag_shows_usage(capsys):
    code = cli.main(["--help"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Usage: study" in captured.out


def test_help_command_without_target(capsys):
    code = cli.main(["help"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Usage: study" in captured.out


def test_list_outputs_command_table(capsys):
    code = cli.main(["list"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Available commands:" in captured.out
    # The table should include the known command names.
    assert "transcribe-video" in captured.out
    assert "markdown-to-pdf" in captured.out


def test_help_known_command(capsys):
    code = cli.main(["help", "transcribe-video"])
    captured = capsys.readouterr()
    assert code == 0
    assert "transcribe-video" in captured.out
    assert "Run `study transcribe-video --help`" in captured.out


def test_help_unknown_command(capsys):
    code = cli.main(["help", "does-not-exist"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Unknown command" in captured.err
    assert "Available commands:" in captured.err


def test_version_command(capsys):
    code = cli.main(["version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "0.0-test"


def test_version_flag(capsys):
    code = cli.main(["--version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "0.0-test"


def test_unknown_command_errors(capsys):
    code = cli.main(["bogus"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Unknown command" in captured.err
    assert "Available commands:" in captured.err
